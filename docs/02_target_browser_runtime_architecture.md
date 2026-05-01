# 02 — Target Browser Runtime Architecture

## Current anchor points

The repo already contains important browser-path infrastructure:

- `scripts/run-local-browser-melo.sh` supervises a MeloTTS HTTP-WAV sidecar and then delegates to `run-local-browser.sh`.
- `src/blink/cli/local_browser.py` builds the FastAPI app, mounts `/client/`, and exposes runtime APIs.
- Existing runtime endpoints include `/api/runtime/expression`, `/api/runtime/voice-metrics`, `/api/runtime/client-media`, `/api/runtime/memory`, `/api/runtime/operator`, and model/runtime inspection endpoints.
- Existing browser vision infrastructure includes `LatestCameraFrameBuffer`, `CameraFeedHealthManager`, `PerceptionBroker`, and the `fetch_user_image` tool.
- Existing voice/runtime processors include `BrainVoiceInputHealthProcessor` and `BrainExpressionVoicePolicyProcessor`.

## Target pipeline

```text
Browser WebRTC media
  -> SmallWebRTC transport
  -> camera frame buffer
  -> pre-STT voice health
  -> STT
  -> post-STT voice health
  -> user aggregator
  -> memory/persona/performance retrieval
  -> LLM
  -> performance plan compiler
  -> Melo speech director
  -> local-http-wav MeloTTS
  -> output transport
  -> assistant aggregator
  -> runtime state + browser UI
```

## New layer: Browser Performance Layer

The browser performance layer should own:

```text
state snapshots
performance events
assistant subtitles
heard transcript
Melo queue health
camera/vision status
interruption status
memory/persona trace
human/eval traces
```

It should not own:

```text
private chain-of-thought
raw credentials
unbounded memory dumps
duplicate media capture
```

## API additions

Preferred:

```text
GET /api/runtime/performance-state
GET /api/runtime/performance-events  # SSE
```

Fallback if SSE is too invasive:

```text
GET /api/runtime/performance-state?since=<event_id>
```

The API should return public-safe, client-renderable facts, not internal prompts.

## UI additions

The `/client/` UI should surface:

```text
Connection: connected/degraded/disconnected
Media: mic/camera permission and track status
Mode: listening/heard/thinking/speaking/looking/interrupted/waiting
Heard: last final transcript
Blink is saying: current subtitle
TTS: Melo ready/degraded, queue depth, first-audio latency
Camera: available/stale/looking/vision result used
Interruption: protected/armed/candidate/accepted/rejected
Memory/persona: used in this reply
```

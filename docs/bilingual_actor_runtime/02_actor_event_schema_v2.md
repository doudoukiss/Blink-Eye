# Actor Event Schema V2 And Bilingual Event Ledger

Phase 02 adds a public-safe actor-event layer beside the existing browser
performance events. The v1 endpoint and `BrowserPerformanceEventBus.emit()`
remain compatible; every v1 event also produces a v2 actor event for UI polling,
debugging, optional traces, replay, and evaluation.

## Canonical Browser Paths

Both primary local browser paths emit the same v2 shape:

- `./scripts/run-local-browser-melo.sh`: `profile=browser-zh-melo`,
  `language=zh`, `tts_backend=local-http-wav`,
  `tts_label=local-http-wav/MeloTTS`, `vision_backend=moondream`.
- `./scripts/run-local-browser-kokoro-en.sh`: `profile=browser-en-kokoro`,
  `language=en`, `tts_backend=kokoro`, `tts_label=kokoro/English`,
  `vision_backend=moondream`.

The shared browser client remains:

```text
http://127.0.0.1:7860/client/
```

When browser vision is explicitly disabled, `vision_backend` becomes `none`.

## Live Polling

The legacy endpoint stays schema v1:

```text
GET /api/runtime/performance-events
```

The new actor endpoint uses the same polling shape with schema v2:

```text
GET /api/runtime/actor-events?after_id=0&limit=50
```

Each event includes:

```json
{
  "schema_version": 2,
  "event_id": 1,
  "event_type": "speaking",
  "mode": "speaking",
  "timestamp": "2026-04-27T00:00:00+00:00",
  "profile": "browser-en-kokoro",
  "language": "en",
  "tts_backend": "kokoro",
  "tts_label": "kokoro/English",
  "vision_backend": "moondream",
  "source": "tts",
  "session_id": "session_123",
  "client_id": "client_123",
  "metadata": {
    "context_available": true
  },
  "reason_codes": [
    "tts:speaking"
  ]
}
```

Canonical event types are `connected`, `listening`, `speech_started`,
`partial_heard`, `final_heard`, `thinking`, `looking`, `speaking`,
`interrupted`, `waiting`, `error`, `memory_used`, `persona_plan_compiled`,
`floor_transition`, `degraded`, and `recovered`.

## Privacy Rules

Actor traces are sanitized more strictly than live v1 performance state. Default
v2 traces do not persist raw audio, raw images, SDP, ICE candidates, secrets,
credentials, hidden prompts, full messages, full prompts, raw text, or unbounded
transcripts. Unsafe metadata keys are omitted. Unsafe string values are omitted.
Nested objects, lists, strings, reason codes, and total retained events are
bounded.

Sanitizer omissions are reported as public reason codes such as
`actor_metadata:unsafe_key_omitted` or `actor_metadata:unsafe_value_omitted`.

## Optional JSONL Tracing

Tracing is off by default. Enable it with either:

```bash
BLINK_LOCAL_ACTOR_TRACE=1 ./scripts/run-local-browser-melo.sh
./scripts/run-local-browser-kokoro-en.sh --actor-trace
```

Disable it explicitly with:

```bash
./scripts/run-local-browser-melo.sh --no-actor-trace
```

Override the trace directory with:

```bash
BLINK_LOCAL_ACTOR_TRACE_DIR=/tmp/blink-actor-traces ./scripts/run-local-browser-kokoro-en.sh
./scripts/run-local-browser-melo.sh --actor-trace-dir /tmp/blink-actor-traces
```

The default directory is:

```text
artifacts/actor_traces/
```

Trace files are named:

```text
actor-trace-{utc_stamp}-{profile}-{run_id}.jsonl
```

Each process writes at most 10,000 actor events. If that limit is reached, one
final `degraded` event with `trace.limit_reached` is written, then trace writing
stops.

## Replay

Replay does not start browser media, camera, model, STT, or TTS runtime code. It
only reads public-safe JSONL actor events and reconstructs mode transitions and
counts:

```bash
scripts/evals/replay-actor-trace.py artifacts/actor_traces/actor-trace-*.jsonl
```

The replay summary reports event count, mode timeline, mode counts, event type
counts, profile/language/TTS/vision labels seen, first and last timestamps, and
trace safety violations.

## Compatibility

`BrowserPerformanceEventBus.emit()` still returns a v1
`BrowserPerformanceEvent`, and `/api/runtime/performance-events` remains schema
v1. Actor events are an additive v2 surface backed by the same event bus.

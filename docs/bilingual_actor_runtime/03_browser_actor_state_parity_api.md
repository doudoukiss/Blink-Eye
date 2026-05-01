# Browser Actor State Parity API

Phase 03 adds one schema-v2 actor-state snapshot for browser UI and diagnostics.
It is additive: existing schema-v1 performance state and event endpoints keep
their current shapes.

## Endpoint

```text
GET /api/runtime/actor-state
```

The endpoint returns one public-safe live snapshot with:

- profile, language, mode, and TTS label
- WebRTC media, microphone, camera, and vision state
- protected playback and interruption state
- active-listening, speech, conversation floor, memory/persona, degradation,
  and last actor event
- bounded live-only text under `live_text`

`GET /api/runtime/performance-state` remains schema v1. `GET
/api/runtime/actor-events` remains the schema-v2 event polling surface from
Phase 02.

## Bilingual Parity

Both primary browser launchers use the same actor-state schema:

- `browser-zh-melo`: `language=zh`, `tts.backend=local-http-wav`,
  `tts.label=local-http-wav/MeloTTS`, `vision.backend=moondream`.
- `browser-en-kokoro`: `language=en`, `tts.backend=kokoro`,
  `tts.label=kokoro/English`, `vision.backend=moondream`.

Client code should branch only for labels or localized copy. State structure is
the same across both profiles.

## Live Text And Trace Safety

`live_text` is a live UI-only section. It may contain bounded user-visible
strings such as a displayed partial transcript or assistant subtitle when a
public runtime surface explicitly provides them. It defaults to `null` values.

Persistent actor traces remain governed by `ActorEventV2`; they store bounded
counts and reason codes by default, not raw transcript, raw audio, raw images,
SDP, ICE candidates, credentials, hidden prompts, or full messages.

## Degradation

`degradation.state` is one of:

- `ok`
- `degraded`
- `error`

Before a client joins, disconnected camera or unreported media is not treated as
a degradation and reports `runtime:waiting_for_client`. During an active browser
session, explicit microphone/media errors report `error`; stale or stalled
camera, unavailable expected vision, speech stale drops, or a degraded actor
event report `degraded`.

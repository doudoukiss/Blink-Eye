# Phase 11: Browser Actor Surface v2

Actor Surface v2 is the default browser overlay for the two primary local
browser paths:

- `./scripts/run-local-browser-melo.sh` (`browser-zh-melo`, Chinese, MeloTTS,
  Moondream)
- `./scripts/run-local-browser-kokoro-en.sh` (`browser-en-kokoro`, English,
  Kokoro, Moondream)

Both continue to use the same client entrypoint:

```text
http://127.0.0.1:7860/client/
```

## Enablement

The surface is enabled by default. It can be disabled for rollback or UI
comparison:

```bash
BLINK_LOCAL_ACTOR_SURFACE_V2=0 ./scripts/run-local-browser-melo.sh
blink-local-browser --no-actor-surface-v2
```

The runtime publishes the flag through `/api/runtime/client-config.js` as
`actor_surface_v2_enabled`. The client renders the v2 actor surface unless that
flag is explicitly `false`.

## Public Data Sources

The overlay uses these public-safe APIs as the primary product surface:

- `/api/runtime/actor-state`
- `/api/runtime/actor-events`

It still polls these v1 endpoints for fallback and advanced diagnostics:

- `/api/runtime/performance-state`
- `/api/runtime/performance-events`

The `/api/runtime/performance-state` payload remains schema v1. The actor
surface reads schema v2 state from `/api/runtime/actor-state`.

## Panels

The same vanilla JS component renders both profiles. Labels are localized:

- English: `Heard`, `Blink is saying`, `Looking`, `Used memory/persona`,
  `Interruption`
- Chinese: `听到`, `Blink 正在说`, `正在看`, `使用的记忆/风格`, `打断`

The compact top status shows profile, language, TTS label, camera/Moondream
state, actor mode, interruption protection/arming, and degradation state.

The panels show:

- Heard: active-listening phase, bounded live text, topics, constraints.
- Blink is saying: current subtitle, speech-director mode, queue depth, stale
  drop count, TTS backend.
- Looking: camera scene state, frame freshness, frame sequence/age, grounding
  mode, and whether this answer used vision.
- Used memory/persona: selected-memory count, persona-reference count, memory
  effect, and compact style summary.
- Interruption: protected/armed/adaptive state, echo risk, last decision, and
  false interruption count.

## Debug Timeline

The `Debug timeline` drawer is collapsed by default. The browser keeps at most
50 actor events and renders the latest 12. It displays only public event fields:
event id, event type, mode, timestamp, source, reason-code categories, and safe
metadata counts/labels.

The drawer does not render metadata keys containing prompt, message,
transcript, audio, image, SDP, ICE, token, secret, candidate, credential, or raw.

## Privacy Boundary

Live UI may show bounded transcript/subtitle strings from `actor_state.live_text`
or current RTVI callbacks. Persistent actor traces remain governed by the
ActorEventV2 sanitizer and do not include raw audio, raw images, SDP, ICE,
secrets, hidden prompts, full messages, or unbounded transcript text.

## Packaging

Edit the source asset:

```text
web/client_src/src/assets/blink-expression-panel.js
```

Then package it:

```bash
cd web/client_src && npm run build
```

The build can copy the browser client workspace into generated package assets
without changing the `/client/` route or script order.

# Current Project Snapshot

This snapshot describes the maintained public source tree after the local-first
runtime, bilingual browser actor surface, and source-first cleanup landed.

## Primary Browser Runtime Paths

Blink has two equal primary browser/WebRTC product paths:

```bash
./scripts/run-local-browser-melo.sh       # Chinese, MeloTTS via local-http-wav
./scripts/run-local-browser-kokoro-en.sh  # English, Kokoro
```

Then open:

```text
http://127.0.0.1:7860/client/
```

The Chinese path starts or reuses the MeloTTS HTTP-WAV sidecar, launches the
browser/WebRTC runtime, enables browser media, and defaults browser vision from
`BLINK_LOCAL_BROWSER_VISION`. The English path uses the same browser/WebRTC
runtime with Kokoro and camera/Moondream available by default. Protected
playback remains the safer baseline; adaptive barge-in should be enabled only
when browser/WebRTC echo health says it is safe.

## Native Kokoro role

Native English Kokoro is no longer the primary UX path. It remains useful for backend isolation:

```text
mic -> STT -> LLM -> TTS
```

It should not be used to judge the final voice + camera UX because PyAudio on Mac speakers lacks browser/WebRTC echo cancellation and can cause self-interruption when barge-in is enabled.

## Current Repo Integration Points

The current project already contains a substantial browser performance-state foundation. Build on these files instead of replacing them:

- scripts/run-local-browser-melo.sh
- scripts/run-local-browser.sh
- src/blink/cli/local_browser.py
- src/blink/interaction/performance_events.py
- src/blink/interaction/browser_state.py
- src/blink/interaction/active_listening.py
- src/blink/interaction/barge_in.py
- src/blink/interaction/camera_presence.py
- src/blink/brain/persona/performance_plan.py
- src/blink/brain/speech_director.py
- src/blink/brain/memory_v2/use_trace.py
- src/blink/brain/evals/browser_perf_bench.py
- web/client_src/src/assets/blink-expression-panel.js
- web/client_src/src/assets/blink-operator-workbench.js
- web/client_src/build.mjs
- tests/test_local_workflows.py

## Existing useful foundations

- Browser runtime endpoints already include `/api/runtime/performance-state`, `/api/runtime/performance-events`, `/api/runtime/client-media`, `/api/runtime/memory`, `/api/runtime/expression`, and behavior-control endpoints.
- `src/blink/interaction/` already contains browser performance events, browser state, active listening, barge-in, and camera-presence helpers.
- `web/client_src/src/assets/blink-expression-panel.js` already displays mode, TTS, mic/camera, interruption, echo, speech, active-listening, camera-grounding, heard text, topics, constraints, subtitle, and memory/persona details.
- Browser client assets live in `web/client_src/src`: authored Blink overlays
  plus vendored browser runtime assets loaded directly by `/client/`. Generated
  package copies can be rebuilt with `web/client_src/build.mjs`, but generated
  `client_dist` output is not committed.
- The brain runtime already has rich memory/persona/expression/voice-policy
  surfaces. The current ambition is to make these surfaces **behaviorally
  consequential** rather than just visible.

## Current Direction

The runtime direction is an actor system whose public behavior can be inspected,
replayed, and improved from local evidence:

```text
performance state -> actor event ledger -> conversation floor controller -> active listener -> speech director -> camera scene state -> persona/memory performance compiler -> evaluated browser actor surface
```

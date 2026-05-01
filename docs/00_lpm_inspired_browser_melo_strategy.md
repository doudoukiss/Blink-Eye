# 00 — LPM-Inspired Strategy for Primary Browser Paths

## Decision

The primary browser/WebRTC interaction paths are now:

```bash
./scripts/run-local-browser-melo.sh
./scripts/run-local-browser-kokoro-en.sh
```

`browser-zh-melo` is the Chinese MeloTTS path with camera, client media state,
existing runtime APIs, and a mounted `/client/` UI. `browser-en-kokoro` is the
equal primary English Kokoro browser/WebRTC path with camera available by default. Native
English Kokoro remains useful, but only as a backend isolation lane.

## Why this changes the plan

The earlier native plan was correct architecturally but wrong as the product focus after the PyAudio self-interruption diagnosis. On laptop speakers, native PyAudio does not get the browser/WebRTC echo-cancellation pipeline. Forced native barge-in caused Blink to hear its own Kokoro output and interrupt itself. The browser paths should therefore carry the primary UX work.

## LPM transfer

LPM's useful concept is not “build a realistic avatar now.” The useful concept is a performance layer. Conversation is not only speech; it includes listening, waiting, reacting, timing, interruption, and long-horizon consistency.

Translate LPM concepts into Blink browser engineering:

| LPM concept | Blink browser equivalent |
|---|---|
| speaking/listening/idle labels | browser interaction state: listening, thinking, speaking, looking, interrupted, waiting |
| speak audio vs listen audio | assistant Melo output vs user WebRTC input |
| online runtime states | WebRTC session states and performance event bus |
| boundary-aligned updates | speech chunk boundaries, UI updates, interruptible turns |
| controlled lookahead | keep Melo output queue short and cancellation-safe |
| multi-reference identity | multi-reference persona examples and memory traces |
| LPM-Bench | Blink Browser Perf Bench |

## Core architecture principle

The browser is now the product shell and media owner. The server remains the intelligence/runtime owner. Do not build a second media stack. Instead, expose performance state and metrics from the server and render them in the browser UI.

## Non-goals for this pass

- Do not optimize the native Kokoro path as the main UX.
- Do not add a second browser media capture implementation.
- Do not enable barge-in by default without echo-safe validation.
- Do not add a realistic avatar yet.
- Do not add another hidden memory system without visible behavioral effects.

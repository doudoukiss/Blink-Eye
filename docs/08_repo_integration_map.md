# 08 — Repo integration map

## Browser runtime

`src/blink/cli/local_browser.py` is currently the main integration file. It wires the browser transport, STT, LLM, TTS, camera buffer, performance event bus, runtime endpoints, behavior controls, and memory/persona APIs.

Do not keep adding unbounded complexity to this file forever. This upgrade should gradually move logic into dedicated `src/blink/interaction/` modules while preserving existing routes.

## Interaction modules

Current modules:

```text
src/blink/interaction/performance_events.py
src/blink/interaction/browser_state.py
src/blink/interaction/active_listening.py
src/blink/interaction/barge_in.py
src/blink/interaction/camera_presence.py
```

New modules proposed:

```text
src/blink/interaction/actor_events.py
src/blink/interaction/floor_control.py
src/blink/interaction/adaptive_interruption.py
src/blink/interaction/speech_performance.py
src/blink/interaction/privacy.py
src/blink/interaction/avatar_adapter.py
```

## Browser UI

Current UI source is asset-based:

```text
web/client_src/src/assets/blink-expression-panel.js
web/client_src/src/assets/blink-operator-workbench.js
web/client_src/build.mjs
```

The current product direction is to improve the main actor surface using the
source asset pipeline first. A frontend framework migration is optional and
should stand on its own when it becomes justified.

## Brain/persona/memory

Relevant existing surfaces:

```text
src/blink/brain/persona/
src/blink/brain/memory_v2/
src/blink/brain/context/
src/blink/brain/processors.py
src/blink/brain/speech_director.py
```

The performance compiler should reuse these surfaces and add inspectable behavior effects rather than creating another hidden memory/personality stack.

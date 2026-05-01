# Repository integration map

## Launchers and profiles

```text
scripts/run-local-browser-melo.sh
scripts/run-local-browser-kokoro-en.sh
scripts/run-local-browser.sh
config/local_runtime_profiles.json
```

## Browser runtime

```text
src/blink/cli/local_browser.py
src/blink/web/smallwebrtc_ui.py
src/blink/transports/smallwebrtc/
```

## Interaction layer

```text
src/blink/interaction/performance_events.py
src/blink/interaction/browser_state.py
src/blink/interaction/active_listening.py
src/blink/interaction/barge_in.py
src/blink/interaction/camera_presence.py
```

Expected new/expanded modules:

```text
src/blink/interaction/actor_events.py
src/blink/interaction/actor_state.py
src/blink/interaction/floor.py
src/blink/interaction/webrtc_audio_health.py
src/blink/interaction/avatar_adapter_contract.py
```

## Brain/persona/memory

```text
src/blink/brain/speech_director.py
src/blink/brain/processors.py
src/blink/brain/persona/
src/blink/brain/memory_v2/
src/blink/brain/memory_persona_ingestion.py
```

## Evaluation

```text
src/blink/brain/evals/browser_perf_bench.py
src/blink/brain/evals/bilingual_actor_bench.py
scripts/eval-bilingual-actor-bench.sh
evals/bilingual_actor_bench/
```

## Browser UI

```text
web/client_src/src/
web/client_src/src/assets/blink-expression-panel.js
web/client_src/src/assets/blink-operator-workbench.js
generated browser package assets, when rebuilt locally
```

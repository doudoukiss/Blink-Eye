# Repository integration map

Read these files first before implementing V3 work:

## Launchers and local runtime

- `scripts/run-local-browser-melo.sh`
- `scripts/run-local-browser-kokoro-en.sh`
- `scripts/run-local-browser.sh`
- `src/blink/cli/local_browser.py`

## Actor runtime

- `src/blink/interaction/actor_events.py`
- `src/blink/interaction/actor_state.py`
- `src/blink/interaction/floor.py`
- `src/blink/interaction/barge_in.py`
- `src/blink/interaction/webrtc_audio_health.py`
- `src/blink/interaction/active_listening.py`
- `src/blink/interaction/camera_presence.py`
- `src/blink/interaction/avatar_adapter_contract.py`

## Brain, persona, memory, speech

- `src/blink/brain/processors.py`
- `src/blink/brain/speech_director.py`
- `src/blink/brain/persona/performance_plan.py`
- `src/blink/brain/persona/compiler.py`
- `src/blink/brain/persona/reference_bank.py`
- `src/blink/brain/memory_v2/continuity_trace.py`
- `src/blink/brain/memory_v2/retrieval.py`
- `src/blink/brain/scene_world_state.py`

## Browser UI

- `web/client_src/src/assets/blink-expression-panel.js`
- `web/client_src/src/assets/blink-operator-workbench.js`
- `web/client_src/src/assets/blink-media-autoplay.js`

## Evaluation and tests

- `src/blink/brain/evals/bilingual_actor_bench.py`
- `scripts/eval-bilingual-actor-bench.sh`
- `scripts/evals/replay-actor-trace.py`
- `tests/test_bilingual_actor_bench.py`
- `tests/test_actor_release_gate.py`
- `tests/test_browser_actor_surface_contract.py`
- `tests/test_conversation_floor_controller.py`
- `tests/test_speech_performance_director_dual_tts.py`
- `tests/test_memory_persona_performance.py`

# File Map

This map is for engineers entering the cleaned public source tree. It focuses
on source code, required configuration, and maintained documentation. Generated
outputs are intentionally absent.

## Runtime Core

| Path | Role |
| --- | --- |
| `src/blink/frames/` | Runtime frame types and protobuf schema. |
| `src/blink/processors/` | `FrameProcessor` base classes, aggregators, filters, and runtime processors. |
| `src/blink/pipeline/` | Pipeline, task, runner, and lifecycle orchestration. |
| `src/blink/transports/` | Local audio, browser/WebRTC, WebSocket, Daily, LiveKit, and related transport boundaries. |
| `src/blink/services/` | STT, TTS, LLM, vision, avatar, and provider integrations. |
| `src/blink/serializers/` | Wire-format serialization for WebSocket and telephony protocols. |
| `src/blink/observers/` | Non-mutating runtime observers and metrics hooks. |

## Local Product Surface

| Path | Role |
| --- | --- |
| `src/blink/cli/local_common.py` | Local defaults, language routing, prompt helpers, and shared CLI utilities. |
| `src/blink/cli/local_chat.py` | Terminal text chat entry point. |
| `src/blink/cli/local_voice.py` | Native mic/speaker voice loop and macOS camera-helper integration. |
| `src/blink/cli/local_browser.py` | Browser/WebRTC runtime, public APIs, actor state, memory/operator surfaces, and `/client/` mounting. |
| `src/blink/cli/local_doctor.py` | Local environment diagnostics for profiles, dependencies, models, and browser client assets. |
| `config/local_runtime_profiles.json` | Product-behavior defaults for primary local browser/native profiles. |
| `env.local.example` | Local override template for ports, devices, sidecars, and optional secrets. |

## Brain, Memory, And Behavior

| Path | Role |
| --- | --- |
| `src/blink/brain/` | Brain runtime, typed memory, continuity graph, planning, evaluations, behavior controls, and embodied action policy. |
| `src/blink/interaction/` | Public interaction state for actor events, floor state, active listening, and WebRTC audio health. |
| `schemas/` | JSON schemas for actor, performance, memory, camera, floor, and release-gate payloads. |
| `evals/` | Deterministic seed cases and human rating forms for browser and bilingual actor evaluation. |

## Browser UI

| Path | Role |
| --- | --- |
| `web/client_src/src/` | Browser client workspace served at `/client/`, including authored Blink overlays and vendored runtime assets. |
| `web/client_src/build.mjs` | Local build/copy script that can regenerate `src/blink/web/client_dist/` when package assets are needed. |
| `src/blink/web/smallwebrtc_ui.py` | Static UI mounting and fallback to browser client assets in a source checkout. |

Generated `src/blink/web/client_dist/` is not committed.

## Local TTS And Native Helpers

| Path | Role |
| --- | --- |
| `local_tts_servers/melotts_http_server.py` | Repo-owned MeloTTS HTTP-WAV sidecar server. |
| `local_tts_servers/melotts_reference.py` | Bootstrap/prefetch helpers for the isolated MeloTTS reference environment. |
| `docs/MeloTTS-reference/` | Requirements and operator notes for the isolated MeloTTS environment. |
| `native/macos/BlinkCameraHelper/` | Source for the macOS camera-permission helper. Build output is ignored. |

## Scripts And Proof Lanes

| Path | Role |
| --- | --- |
| `scripts/bootstrap-blink-mac.sh` | Local profile bootstrap entry point. |
| `scripts/run-blink-chat.sh` | Default local text workflow. |
| `scripts/run-local-browser-melo.sh` | Primary Chinese browser path. |
| `scripts/run-local-browser-kokoro-en.sh` | Primary English browser path. |
| `scripts/test-brain-core.sh` | Canonical brain-core proof lane. |
| `scripts/test-embodied-core.sh` | Canonical embodied/perception proof lane. |
| `scripts/eval-bilingual-actor-bench.sh` | Browser actor runtime release gate. |

## Documentation

| Path | Role |
| --- | --- |
| `README.md` | Public project overview. |
| `LOCAL_DEVELOPMENT.md` | Engineering setup and workflow guide. |
| `LOCAL_CAPABILITY_MATRIX.md` | Local vs external capability matrix. |
| `tutorial.md` | Architecture tutorial. |
| `docs/README.md` | Documentation index. |
| `docs/roadmap.md` | Current ambition and product direction. |
| `docs/chinese-conversation-adaptation.md` | Authoritative Chinese-local adaptation record. |
| `docs/bilingual_actor_runtime/` | Browser actor runtime and bilingual release-gate documentation. |

## Generated Or Disposable Output

These paths are intentionally absent from the public source tree and ignored if
created locally:

- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`
- `artifacts/`
- `src/blink/web/client_dist/`
- `native/macos/BlinkCameraHelper/build/`
- provider/model caches and local sidecar checkouts

Required runtime assets that code loads directly, including the browser client
assets under `web/client_src/src` and packaged model assets, are retained.

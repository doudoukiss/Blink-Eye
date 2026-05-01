# Blink

Blink is a local-first framework for real-time voice and multimodal
conversational assistants. It installs as `blink-ai`, imports as `blink`, and
is organized around a frame pipeline that can run text, speech, browser/WebRTC,
vision, memory, and optional service integrations through the same runtime
model.

The project is built for engineers who want a serious local development
surface before they add hosted providers or production transports. The default
path is a MacBook Pro workflow using Python 3.12, `uv`, Ollama, local STT, and
local TTS. Provider integrations remain available, but they are optional rather
than required for a first run.

## What Makes Blink Different

- **Frame-based runtime:** audio, text, video, transcription, interruption, and
  lifecycle events move through composable `FrameProcessor` pipelines.
- **Local-first product path:** terminal chat, native voice, and browser/WebRTC
  voice can all be exercised from local scripts.
- **Bilingual browser surface:** Chinese and English browser profiles are
  treated as equal primary product lanes:
  `browser-zh-melo` for Chinese MeloTTS via `local-http-wav`, and
  `browser-en-kokoro` for English Kokoro.
- **Observable actor state:** the browser runtime exposes public-safe state for
  listening, thinking, speaking, looking, interruption, camera honesty,
  memory/persona use, and degradation.
- **Memory and continuity:** Blink includes typed local memory, continuity
  evaluation lanes, behavior controls, and bounded public traces designed for
  inspection without exposing secrets, prompts, raw audio, or raw images.
- **Optional integrations:** hosted STT, TTS, LLM, WebRTC, telephony, avatar,
  and observability integrations live beside the local path instead of
  replacing it.

## Quick Start

Install the local text profile, start Ollama, and run a one-terminal chat:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
ollama serve
./scripts/run-blink-chat.sh
```

The default local language is Simplified Chinese. English remains supported:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-chat.sh
```

For the primary browser paths:

```bash
# Chinese browser voice with MeloTTS via local-http-wav
./scripts/bootstrap-melotts-reference.sh
./scripts/run-melotts-reference-server.sh
./scripts/run-local-browser-melo.sh

# English browser voice with Kokoro
./scripts/run-local-browser-kokoro-en.sh
```

Then open:

```text
http://127.0.0.1:7860/client/
```

For a focused local health check:

```bash
./scripts/doctor-blink-mac.sh --profile text
```

## Architecture

Blink is not a single chatbot loop. It is a reusable real-time pipeline system:

```text
input transport -> processor -> processor -> processor -> output transport
```

Core concepts:

- `Frame`: typed runtime data and control messages.
- `FrameProcessor`: composable async processing units.
- `Pipeline`: ordered processor graph.
- `PipelineTask`: lifecycle, cancellation, heartbeat, and observer wrapper.
- `Transport`: external I/O boundary for local audio, browser/WebRTC,
  WebSocket, Daily, LiveKit, and other integrations.
- `Service`: STT, TTS, LLM, vision, avatar, and provider adapters.

Start with [`tutorial.md`](./tutorial.md) for the mental model and
[`docs/FILE_MAP.md`](./docs/FILE_MAP.md) for the current source layout.

## Source-First Repository State

This checkout is intentionally kept to source, required configuration, and
runtime assets that the code loads directly. Cache, build, runtime, and
disposable artifacts are excluded.

Important consequences:

- The browser client workspace lives in
  [`web/client_src/src`](./web/client_src/src). It contains authored Blink
  overlays plus the vendored browser runtime assets needed for `/client/`.
- Local/package browser assets can be regenerated with:

  ```bash
  node web/client_src/build.mjs
  ```

- Generated copies such as `src/blink/web/client_dist/`, `artifacts/`,
  native helper builds, caches, and virtual environments should not be
  committed.
- Runtime model assets that code loads directly, such as the bundled Smart Turn
  and VAD ONNX files, and the browser runtime assets loaded by `/client/`, are
  kept because they are required package data.

## Proof Lanes

For most source changes:

```bash
uv sync --python 3.12 --group dev
uv run pytest
uv run ruff check
uv run ruff format --check
```

For brain-core work:

```bash
./scripts/test-brain-core.sh
./scripts/test-brain-core.sh --lane fast
./scripts/test-brain-core.sh --lane proof
./scripts/test-brain-core.sh --lane fuzz-smoke
```

For embodied/perception work:

```bash
./scripts/test-embodied-core.sh
./scripts/test-embodied-core.sh --lane fast
./scripts/test-embodied-core.sh --lane proof
```

For browser actor/runtime changes:

```bash
./scripts/eval-bilingual-actor-bench.sh
```

## Documentation

- [Local development](./LOCAL_DEVELOPMENT.md)
- [Capability matrix](./LOCAL_CAPABILITY_MATRIX.md)
- [User manual](./docs/USER_MANUAL.md)
- [Architecture tutorial](./tutorial.md)
- [Documentation index](./docs/README.md)
- [Roadmap](./docs/roadmap.md)
- [Chinese conversation adaptation](./docs/chinese-conversation-adaptation.md)
- [Security policy](./SECURITY.md)

## Project Direction

Blink’s near-term ambition is to make local voice and browser conversation feel
inspectable, grounded, and dependable: clear turn-taking, honest camera state,
bounded interruption behavior, memory that changes behavior, and public-safe
operator surfaces. The longer-term direction is an adaptive runtime that can
learn from structured local evidence while preserving explicit user control,
privacy boundaries, and reproducible proof lanes.

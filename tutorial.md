# Blink Tutorial

This tutorial explains the repository as a real local development system under the Blink product identity, not as a generic voice-AI brochure.

## 0. Choose A Practical Starting Point

If you just want the fastest reliable entry point, start with `text`:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
ollama serve
./scripts/run-blink-chat.sh
```

The default local interaction language is Simplified Chinese. English remains an explicit fallback.

If you want spoken interaction on your MacBook Pro, choose one of these:

- native mic/speaker voice:
  `./scripts/bootstrap-blink-mac.sh --profile voice`
- browser/WebRTC voice:
  `./scripts/bootstrap-blink-mac.sh --profile browser`
- browser/WebRTC voice with camera preinstalled:
  `./scripts/bootstrap-blink-mac.sh --profile browser --with-vision`

If you do not want model downloads to happen during your first interaction,
prefetch them up front:

```bash
./scripts/bootstrap-blink-mac.sh --profile full --with-vision --with-piper --prefetch-assets
```

The default spoken Chinese bootstrap path still stays on the in-repo Kokoro
backend so native local voice can run directly on an Apple Silicon Mac. That is
the bootstrap path, not the Chinese-quality target. The primary Chinese browser
path, `browser-zh-melo`, uses MeloTTS behind `local-http-wav` with camera
support enabled; the equal primary English browser path, `browser-en-kokoro`,
uses Kokoro with browser camera/Moondream enabled by default and continuous
perception off by default. XTTS and CosyVoice remain optional secondary paths.

The full technical record of that adaptation work lives in
[`docs/chinese-conversation-adaptation.md`](./docs/chinese-conversation-adaptation.md).

The built-in prompt defaults are now split on purpose:

- text chat keeps a developer-friendly Chinese default
- voice and browser use a stricter speech-safe Chinese default

If you set `OLLAMA_SYSTEM_PROMPT`, that shared override replaces both.

Speech is fixed per session in the current local product path. If you start a Chinese session and then tell the assistant to "speak English now", the LLM may change text, but the TTS service will not hot-swap. Start a new English session when you want spoken English.

## 1. Start With The Local Profiles

The repo now has four local bootstrap profiles:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
./scripts/bootstrap-blink-mac.sh --profile voice
./scripts/bootstrap-blink-mac.sh --profile browser
./scripts/bootstrap-blink-mac.sh --profile full
```

If you only want the fastest learning path, start with:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
ollama serve
./scripts/run-blink-chat.sh
```

That path gives you:

- local Ollama inference
- model `qwen3.5:4b`
- a real Blink frame pipeline
- no cloud API keys
- no mic, speaker, browser, or telephony setup
- Simplified Chinese replies by default

If you want the full local voice stack on an MBP:

```bash
./scripts/bootstrap-blink-mac.sh --profile full
ollama serve
./scripts/run-blink-voice.sh
./scripts/run-blink-browser.sh
```

If you want all selected local model downloads to happen before you start using
the product, run bootstrap with `--prefetch-assets` or run the prefetch script
separately:

```bash
./scripts/bootstrap-blink-mac.sh --profile full --with-vision --with-piper --prefetch-assets
./scripts/prefetch-blink-assets.sh --profile browser --with-vision
./scripts/prefetch-blink-assets.sh --profile voice --tts-backend piper
```

The `browser-zh-melo` launcher assumes you want the Chinese Melo-backed camera
path. If you want to preinstall the optional vision stack before first launch:

```bash
./scripts/bootstrap-blink-mac.sh --profile browser --with-vision
ollama serve
./scripts/run-blink-browser.sh
```

## 2. What This Repository Actually Is

The runtime substrate is Blink's frame-based real-time engine. The important unit is not a chatbot function. It is a frame pipeline.

In practice:

- inputs become `Frame` objects
- `FrameProcessor` instances transform or route those frames
- a `Pipeline` connects processors
- a `PipelineTask` runs the pipeline and manages lifecycle, cancellation, and metrics

The local workflows added in this repo are simply different ways of exercising that same architecture:

- terminal chat
- native English voice
- browser/WebRTC voice
- browser camera inspection through local vision
- browser/native voice with shared Blink brain runtime
- browser/native voice with optional bounded robot embodiment

The practical defaults are:

- Chinese local text: Ollama + Chinese system instruction
- Native English speech: MLX Whisper (`mlx-community/whisper-medium-mlx`) + Kokoro
- Chinese browser speech bootstrap: MLX Whisper + Kokoro
- Chinese browser speech quality path: MLX Whisper + MeloTTS via `local-http-wav`
- English browser speech and camera grounding: MLX Whisper + Kokoro + Moondream through
  `./scripts/run-local-browser-kokoro-en.sh`
- English fallback speech: MLX Whisper + Kokoro
- English fallback TTS alternative: Piper
- daily-use brain path: browser/WebRTC with equal primary `browser-zh-melo` and
  `browser-en-kokoro` profiles, `BrainRuntime`, typed SQLite memory, bounded
  `brain_*` tools, and bounded `robot_head_*` actions

## 3. The Smallest Useful Mental Model

### Frames

Frames are the units that move through the system. Examples:

- user text
- transcriptions
- LLM token output
- audio chunks
- images
- interruptions
- lifecycle signals like start, end, and cancel

Read: `src/blink/frames/frames.py`

### Processors

Processors consume frames, do work, and push new frames.

Read: `src/blink/processors/frame_processor.py`

Examples:

- context aggregation
- LLM invocation
- TTS
- STT
- transport input and output
- logging and observability

### Pipelines

A pipeline is an ordered chain of processors:

```text
input -> processor -> processor -> processor -> output
```

Read: `src/blink/pipeline/pipeline.py`

### Pipeline Tasks

`PipelineTask` is the runtime wrapper that starts the flow, handles shutdown, and injects lifecycle frames.

Read: `src/blink/pipeline/task.py`

## 4. The Three Local Runtime Paths

### Text path

The text path lives in `src/blink/cli/local_chat.py`.

Its pipeline is:

```text
LLMContextFrame -> OllamaLLMService -> ConsoleLLMPrinter
```

This is the shortest path for debugging context handling, frame flow, and local LLM execution.

### Native English voice path

The native voice path lives in `src/blink/cli/local_voice.py`.

Its pipeline is:

```text
LocalAudioTransport.input()
-> local STT
-> LLM user aggregation
-> OllamaLLMService
-> local TTS
-> LocalAudioTransport.output()
-> LLM assistant aggregation
```

Default local stack:

- STT: MLX Whisper, with the English default pinned to `mlx-community/whisper-medium-mlx`
- LLM: Ollama `qwen3.5:4b`
- TTS: Kokoro
- turn handling: Silero VAD + Smart Turn v3

Native voice intentionally does not use camera, Moondream, browser/WebRTC, or
MeloTTS. Use browser voice for camera grounding and the Chinese MeloTTS path.

There is one separate macOS-only helper path for English camera grounding
without browser/WebRTC:

```bash
./scripts/bootstrap-blink-mac.sh --profile voice --with-vision
./scripts/build-macos-camera-helper.sh
./scripts/run-local-voice-macos-camera-en.sh
```

That path still uses English Kokoro voice. `BlinkCameraHelper.app` owns macOS
camera permission, writes a low-FPS RGB snapshot cache, and Moondream reads the
latest fresh frame only when `fetch_user_image` is called.

The native voice wrappers default to protected playback. Native PyAudio does not
provide browser/WebRTC echo cancellation, so open speakers can otherwise leak
Blink's own voice back into the microphone and cut answers short. Use
`--allow-barge-in` only with headphones or another echo-safe setup.

Supported fallback stack:

- STT: Faster Whisper
- TTS: XTTS, local HTTP WAV, or Piper

### Browser voice path

The browser/WebRTC path lives in `src/blink/cli/local_browser.py`.

It uses the same local STT, LLM, and TTS stack as native voice, but swaps
`LocalAudioTransport` for `SmallWebRTCTransport` and serves the repo-owned Blink
browser UI at `http://127.0.0.1:7860/client/`.

The canonical launcher `./scripts/run-blink-browser.sh` now defaults to the
Chinese MeloTTS browser path with camera support enabled. The lower-level
browser CLI still exposes explicit `--vision` and `--tts-backend ...` controls
when you need to bypass those defaults.

## 5. Files Worth Reading First

If you want a fast mental model, read in this order:

1. `src/blink/cli/local_common.py`
2. `src/blink/cli/local_chat.py`
3. `src/blink/cli/local_voice.py`
4. `src/blink/cli/local_browser.py`
5. `src/blink/processors/aggregators/llm_response_universal.py`
6. `src/blink/pipeline/task.py`
7. `src/blink/runner/run.py`

Then look at the curated local examples:

- `examples/local/01-ollama-terminal-chat.py`
- `examples/local/02-local-native-voice.py`
- `examples/local/03-local-browser-voice.py`
- `examples/local/04-local-transcription.py`
- `examples/local/05-local-vision-moondream.py`

Then read the three Chinese reference prototypes in `docs/0415`.

## 6. How To Change The Local Behavior

### Change the default local model

Edit `.env`:

```dotenv
OLLAMA_MODEL=qwen3.5:4b
```

Or pass a flag:

```bash
./scripts/run-blink-chat.sh --model qwen3.5:4b
```

### Change the STT or TTS backend

Edit `.env`:

```dotenv
BLINK_LOCAL_STT_BACKEND=mlx-whisper
BLINK_LOCAL_TTS_BACKEND=kokoro
```

Or override at runtime:

```bash
./scripts/run-blink-voice.sh --stt-backend whisper --tts-backend piper
./scripts/run-blink-browser.sh --tts-backend local-http-wav
```

If you want Piper installed as a local English fallback, add it during bootstrap:

```bash
./scripts/bootstrap-blink-mac.sh --profile voice --with-piper
./scripts/bootstrap-blink-mac.sh --profile browser --with-piper
```

If you want Piper to be your default instead of a command-line override, set:

```dotenv
BLINK_LOCAL_TTS_BACKEND=piper
BLINK_LOCAL_TTS_VOICE_EN=en_US-ryan-high
```

If you want the older English-first local flow instead of the Chinese default:

```dotenv
BLINK_LOCAL_LANGUAGE=en
BLINK_LOCAL_TTS_BACKEND=kokoro
BLINK_LOCAL_TTS_VOICE_EN=af_heart
```

If you want a local Mandarin HTTP TTS service instead of XTTS:

```dotenv
BLINK_LOCAL_TTS_BACKEND=local-http-wav
BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001
```

The primary recommended browser way to satisfy that contract in this repo is
MeloTTS:

```bash
./scripts/bootstrap-melotts-reference.sh
./scripts/run-local-browser-melo.sh
```

If you do have an XTTS-compatible server available, enable it explicitly:

```dotenv
BLINK_LOCAL_TTS_BACKEND=xtts
BLINK_LOCAL_XTTS_BASE_URL=http://127.0.0.1:8000
BLINK_LOCAL_TTS_VOICE_ZH=
```

Voice resolution order in the local runtime is:

1. CLI `--tts-voice`
2. `BLINK_LOCAL_TTS_VOICE_ZH` or `BLINK_LOCAL_TTS_VOICE_EN`
3. `BLINK_LOCAL_TTS_VOICE`
4. backend/language default

That split matters because a single generic voice override can make English sessions inherit a Chinese voice or vice versa.

### Evaluate speech quality explicitly

Use the built-in evaluation harness when you want reproducible WAV outputs for listening tests:

```bash
./scripts/eval-local-tts.sh
./scripts/eval-local-tts.sh --backend kokoro --all-kokoro-zh-voices
./scripts/eval-local-tts.sh --backend local-http-wav --language zh --voice-zh speaker-a
```

Outputs go to `artifacts/tts-eval/`. This is the repo-owned boundary for manual zh/en comparison.
Future higher-quality speech engines should fit behind the existing `local-http-wav` contract rather
than being copied directly into Blink from a reference folder.

### Run the recommended MeloTTS sidecar

This repo now includes a repo-owned MeloTTS HTTP-WAV server outside `src/blink`:

```bash
./scripts/bootstrap-melotts-reference.sh
./scripts/run-melotts-reference-server.sh
```

Then switch browser Blink onto that path:

```bash
BLINK_LOCAL_TTS_BACKEND=local-http-wav \
BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001 \
./scripts/run-blink-browser.sh
```

Or use the one-command browser wrapper:

```bash
./scripts/run-local-browser-melo.sh
```

`./scripts/run-local-browser-melo.sh` now preserves the same camera-enabled
browser default as `./scripts/run-blink-browser.sh`; it should not silently
drop the session to audio-only browser mode.

### Run the included CosyVoice adapter

This repo now includes a sidecar adapter that exposes Blink's `/tts` contract while proxying to
an external CosyVoice server:

```bash
./scripts/run-local-cosyvoice-adapter.sh
```

Then switch Blink onto that path:

```bash
BLINK_LOCAL_TTS_BACKEND=local-http-wav \
BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001 \
./scripts/run-blink-voice.sh
```

See [CHINESE_TTS_UPGRADE.md](./CHINESE_TTS_UPGRADE.md) for the full Chinese-quality upgrade workflow.

If you want to use a local CosyVoice reference checkout directly:

```bash
./scripts/bootstrap-cosyvoice-reference.sh
./scripts/run-local-browser-cosyvoice.sh
```

That bootstraps an external reference environment, starts the upstream FastAPI
server, starts the Blink adapter, and then runs the browser/WebRTC Blink app
with `local-http-wav`. The reference source is optional and is not committed in
the source-first repository.

That path is optional. For the default runnable MBP flow, stay on `kokoro` and use
the stronger speech-safe prompt plus the improved Chinese normalization. If the
spoken Mandarin accent still feels heavy, the first upgrade path is MeloTTS via
`local-http-wav`, not a larger mandatory dependency in the core repo.

### Select explicit audio devices

```bash
./scripts/run-blink-voice.sh --list-audio-devices
./scripts/run-blink-voice.sh --input-device 1 --output-device 3
```

### Add a custom processor

The cleanest way to experiment is to insert a processor into one of the local CLI pipelines.

For example:

1. create a processor in `src/blink/processors/`
2. import it in `src/blink/cli/local_chat.py`, `local_voice.py`, or `local_browser.py`
3. place it before or after the service you want to inspect
4. rerun the corresponding `run-local-*` command

## 7. What Still Is Not Local

The repo now has real local paths for text, voice, browser, transcription, and optional vision. Some areas still require external services:

- Daily
- LiveKit
- telephony providers
- hosted STT/TTS providers
- hosted realtime speech-to-speech APIs
- hosted video avatars

Use [`LOCAL_CAPABILITY_MATRIX.md`](./LOCAL_CAPABILITY_MATRIX.md) for the exact split.

## 8. Common Commands

Bootstrap:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
```

Interactive chat:

```bash
./scripts/run-blink-chat.sh
```

English fallback:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-chat.sh
```

Native voice:

```bash
./scripts/run-blink-voice.sh
```

Browser voice:

```bash
./scripts/run-blink-browser.sh
```

Browser voice with explicit bootstrap fallback:

```bash
./scripts/run-blink-browser.sh --tts-backend kokoro
```

English browser voice with Kokoro, Moondream, and no Melo sidecar:

```bash
./scripts/run-local-browser-kokoro-en.sh
```

Optional XTTS-compatible server for a higher-quality external Mandarin voice path:

```bash
BLINK_LOCAL_TTS_BACKEND=xtts ./scripts/run-blink-voice.sh
```

Doctor:

```bash
./scripts/doctor-blink-mac.sh --profile full
```

Tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check
```

## 9. Product Direction For This Repo

Treat this repository as the framework itself, with a strong local development shell around it.

The practical stance is:

- terminal chat is the fastest sanity check
- native voice and browser voice are first-class local developer workflows
- telephony and hosted providers are optional integrations
- add extras only when the feature requires them
- prefer explicit frame flow over hidden convenience layers

# Local Capability Matrix

This repository now treats Apple Silicon macOS as a first-class development target. The matrix below separates what is locally runnable today from what still depends on external infrastructure.

## 1. Runs Locally Now

### Core local workflows

| Capability | Local path | Notes |
| --- | --- | --- |
| Terminal chat | `./scripts/run-blink-chat.sh` | Uses Ollama and `qwen3.5:4b`. No audio or browser setup. |
| Terminal chat, hybrid | `./scripts/run-blink-chat-openai.sh` | Keeps the local chat shell but routes only the LLM layer to OpenAI Responses. |
| Native English voice | `./scripts/run-local-voice-en.sh` | Backend-isolation lane for Mac mic, MLX Whisper, Ollama, Kokoro, and native speaker output. Camera, Moondream, browser/WebRTC, and MeloTTS are intentionally off. |
| Native English voice + macOS camera helper | `./scripts/run-local-voice-macos-camera-en.sh` | Backend-plus-helper-camera isolation lane. `BlinkCameraHelper.app` owns macOS camera permission, and Moondream runs only on explicit single-frame image-tool requests. No browser/WebRTC, OpenCV camera capture, or MeloTTS. |
| Native voice, hybrid | `./scripts/run-blink-voice-openai.sh` | Keeps local STT and Kokoro TTS, routes only text generation to OpenAI Responses. |
| Browser Chinese Melo voice | `./scripts/run-local-browser-melo.sh` | Equal primary browser/WebRTC product path for Chinese: MLX Whisper, Ollama, MeloTTS via `local-http-wav`, protected playback, camera enabled, continuous perception off by default. |
| Browser English Kokoro voice | `./scripts/run-local-browser-kokoro-en.sh` | Equal primary browser/WebRTC product path for English: browser/WebRTC media handling with English MLX Whisper, Ollama, Kokoro, protected playback, camera/Moondream enabled by default, and continuous perception off by default. No MeloTTS sidecar. |
| Browser voice, generic | `./scripts/run-blink-browser.sh` | Uses `SmallWebRTCTransport` with the same local STT/LLM/TTS stack and the repo-owned Blink browser UI. Prefer the explicit `browser-zh-melo` or `browser-en-kokoro` wrappers for daily product-path work. |
| Browser voice, hybrid | `./scripts/run-blink-browser-openai.sh` | Keeps browser camera, WebRTC, local STT, and local MeloTTS; routes only the LLM layer to OpenAI Responses. |
| Local diagnostics | `./scripts/doctor-blink-mac.sh` | Checks commands, Python extras, Ollama reachability, audio devices, and local model caches. |

### Local services and processors already supported in this repo

| Area | Local option | Notes |
| --- | --- | --- |
| LLM | Ollama | Default local model is `qwen3.5:4b`. |
| LLM | OpenAI Responses | Optional hybrid demo lane for chat, native voice, and browser. Requires `OPENAI_API_KEY`; wrappers default to `gpt-5.4-mini` unless overridden. |
| STT | MLX Whisper | Recommended default on Apple Silicon. |
| STT | Faster Whisper | Supported as an explicit fallback. |
| TTS | Kokoro | Default bootstrap TTS backend for native local voice and explicit fallback runs. |
| TTS | MeloTTS via `local-http-wav` | Primary recommended Chinese-quality upgrade path, kept outside the main package dependency graph. |
| TTS | Piper | Supported as an explicit fallback for the local English-only workflow. |
| TTS | CosyVoice via `local-http-wav` | Advanced optional sidecar path for stronger external Mandarin speech. |
| Turn handling | Silero VAD + Smart Turn v3 | Smart Turn v3 model is bundled with the package. |
| Vision | Moondream | Optional local extra. First use downloads model weights. |
| Audio transport | LocalAudioTransport | Native mic and speaker I/O via PortAudio/PyAudio. |
| Browser transport | SmallWebRTCTransport | Local browser client served from the repo. |

## 2. Made Locally Runnable By This Local MBP Workflow

These workflows existed as raw framework components or scattered examples before, but are now packaged as first-class local development paths:

- A profile-based bootstrap:
  - `text`
  - `voice`
  - `browser`
  - `full`
- One-command native voice startup with local defaults and device overrides.
- One-command browser/WebRTC startup with the same local backend stack.
- A minimal `env.local.example` instead of starting from the provider-heavy `env.example`.
- A doctor command that reports missing extras, missing `portaudio`, missing Ollama models, and uncached local model assets.
- A curated local example suite in [`examples/local/`](./examples/local/).

## 3. Still External or Infrastructure-Bound

These integrations remain intentionally outside the default local workflow because they require third-party credentials, hosted services, or provider-specific infrastructure:

- Daily rooms and Daily PSTN flows
- LiveKit rooms and tokens
- Telephony providers such as Twilio, Telnyx, Plivo, Exotel, and WhatsApp
- Hosted STT/TTS providers such as Deepgram, Cartesia, ElevenLabs, Google, AssemblyAI, Speechmatics, and similar services
- Hosted LLM and realtime speech APIs such as OpenAI Realtime, Gemini Live, AWS Nova Sonic, Grok voice, and Ultravox
- Hosted video-avatar vendors such as Tavus, HeyGen, Simli, and LemonSlice
- Cloud memory/account-provisioning flows such as Daily room creation, vendor dashboards, and external account setup

## Recommended Local Default

For a MacBook Pro on Apple Silicon, the recommended local stack is:

- Ollama with `qwen3.5:4b`
- MLX Whisper
- Kokoro for text and the lightest native voice bootstrap
- Kokoro in `./scripts/run-local-browser-kokoro-en.sh` for English daily voice
  when browser/WebRTC media and Moondream grounding are desired without MeloTTS
- Kokoro for English native voice without browser/WebRTC or camera
- `BlinkCameraHelper.app` only when testing English native voice with on-demand
  camera grounding outside the browser
- MeloTTS behind `local-http-wav` for `browser-zh-melo` and for real Chinese speech quality tuning
- `LocalAudioTransport` for native audio
- SmallWebRTC with camera enabled for browser testing

Native English Kokoro guardrails are documented in
[`docs/debugging/native_voice_isolation.md`](docs/debugging/native_voice_isolation.md).

For an investor-style hybrid demo, the recommended stack is:

- local STT
- OpenAI Responses for the LLM layer only
- local MeloTTS through `local-http-wav`
- browser/WebRTC with camera enabled
- explicit demo mode for bounded speech-safe answers

Use [`docs/HYBRID_OPENAI_DEMO_RUNBOOK.md`](./docs/HYBRID_OPENAI_DEMO_RUNBOOK.md)
for the launch and pre-demo verification sequence. Pricing changes over time;
check current provider docs before budgeting a live demo.

## Actor Runtime Release Gate

The two primary browser profiles are also evaluated as one bilingual product
surface. Run the deterministic actor bench when changing browser state, actor
events, active listening, speech chunking, camera grounding, interruption,
memory/persona behavior, or browser UI:

```bash
./scripts/eval-bilingual-actor-bench.sh
```

The gate compares `browser-zh-melo` and `browser-en-kokoro`, requires each
quality dimension to stay at least `4.0/5.0`, and fails immediately on hard
blockers such as unsafe trace payloads, hidden camera use, false camera claims,
self-interruption, stale TTS after interruption, memory contradiction, missing
privacy controls, profile regression, or realistic-human avatar capability.

Use [`LOCAL_DEVELOPMENT.md`](./LOCAL_DEVELOPMENT.md) for the concrete install and run commands, and [`tutorial.md`](./tutorial.md) for the repo-level mental model.

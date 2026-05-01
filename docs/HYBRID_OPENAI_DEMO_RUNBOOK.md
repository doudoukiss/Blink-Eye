# Hybrid OpenAI Demo Runbook

This runbook is for the optional Blink hybrid demo lane:

```text
local STT -> OpenAI Responses LLM -> local TTS / MeloTTS
```

The default local lane remains Ollama. Use this path when you want a polished,
low-latency demo while keeping speech, browser, camera, memory, and local
runtime surfaces on your machine.

## What Stays Local

- microphone and browser media capture
- speech recognition
- MeloTTS or other local TTS through `local-http-wav`
- browser/WebRTC runtime
- camera and local vision tooling
- Blink memory, expression state, behavior controls, and operator surfaces

## What Becomes Remote

- only the LLM text-generation call when `openai-responses` is selected

No remote STT or remote TTS is enabled by these wrappers.

## Setup

Set your OpenAI key in the shell or ignored `.env` file:

```bash
export OPENAI_API_KEY=...
```

The wrappers default to `gpt-5.4-mini` for the low-latency demo lane:

```bash
./scripts/run-blink-chat-openai.sh
./scripts/run-blink-voice-openai.sh
./scripts/run-blink-browser-openai.sh
```

For a hero-flow quality check, override the model:

```bash
BLINK_LOCAL_OPENAI_RESPONSES_MODEL=gpt-5.4 ./scripts/run-blink-browser-openai.sh
```

Demo mode is on by default in the wrappers. To disable it:

```bash
BLINK_LOCAL_DEMO_MODE=0 ./scripts/run-blink-browser-openai.sh
```

## Recommended Browser Demo

Use the browser wrapper for the strongest product demo:

```bash
./scripts/run-blink-browser-openai.sh
```

Open:

```text
http://127.0.0.1:7860/client/
```

Allow microphone and camera access.

## Pre-Demo Checks

Run the lightweight LLM-only smoke first:

```bash
./scripts/smoke-hybrid-openai-demo.sh
```

This uses the local chat `--once` lane with `openai-responses`. It proves only
that the OpenAI LLM path, key, model, and demo-mode text settings are usable. It
does not prove STT, TTS, WebRTC, MeloTTS latency, or camera behavior.

Run doctor:

```bash
./scripts/doctor-blink-mac.sh --profile browser --llm-provider openai-responses --demo-mode
```

After the browser server starts, confirm the public stack:

```bash
curl http://127.0.0.1:7860/api/runtime/stack
```

Confirm:

- `llm_provider` is `openai-responses`
- `model` is the model you intend to use
- `demo_mode` is `true`
- STT remains local
- TTS is `local-http-wav` for the Melo path
- `vision_enabled` is true when camera demos are expected

## Manual QA Checklist

Use this before investor-facing use.

| Area | Command or action | Pass criteria | Notes |
| --- | --- | --- | --- |
| Text chat smoke | `./scripts/smoke-hybrid-openai-demo.sh` | One bounded answer returns from `openai-responses`; no markdown-heavy output. | LLM-only, not audio proof. |
| Native voice smoke | `./scripts/run-blink-voice-openai.sh` | Blink listens locally, answers in short spoken Chinese, and TTS remains local. | Use headphones or protected playback defaults. |
| Browser/WebRTC smoke | `./scripts/run-blink-browser-openai.sh` then open `/client/` | Browser connects, microphone works, first answer is spoken, operator workbench loads. | Allow mic/camera permissions. |
| Melo sidecar health | `curl http://127.0.0.1:8001/healthz` and `curl http://127.0.0.1:8001/voices` | Sidecar responds and exposes expected speakers. | The sidecar is an API, not the browser UI. |
| Provider stack | `curl http://127.0.0.1:7860/api/runtime/stack` | `llm_provider=openai-responses`, expected model, `demo_mode=true`, local STT/TTS. | Check this after browser server starts. |
| Camera path | Ask `请描述一下你现在看到的画面` | Blink uses current camera frame or states uncertainty/stale feed; it should not claim camera is unavailable when permission is granted. | Keep scene steady and well-lit. |
| Missing key failure | Temporarily unset `OPENAI_API_KEY` and run an OpenAI wrapper | Wrapper exits before startup with a clear missing-key message. | Do not paste keys into screenshots. |
| Connectivity failure | Block network or use an invalid OpenAI base URL for a test run | Failure is visible and bounded; local STT/TTS/browser setup is not misdiagnosed as broken. | Restore env after the test. |

## Safety And Privacy Notes

The stack endpoint is intentionally public-safe. It must not expose API keys,
auth headers, prompts, raw request payloads, source refs, event ids, DB paths,
or memory internals.

Do not paste pricing tables into this repo. Pricing and service-tier details
change over time; check current provider docs before budgeting a live demo.

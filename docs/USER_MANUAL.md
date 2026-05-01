# Blink Local User Manual

This manual explains how to use the local Blink product path in this
repository.

It is written for someone who wants to run the system, talk to it, and know
which workflow to choose without having to read the engineering notes first.

For the full technical background, see:

- [`LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md)
- [`docs/chinese-conversation-adaptation.md`](./chinese-conversation-adaptation.md)
- [`docs/debugging.md`](./debugging.md)
- [`CHINESE_TTS_UPGRADE.md`](../CHINESE_TTS_UPGRADE.md)

## What This Product Does

This repo provides a local conversational assistant that can run in three main
ways:

- terminal text chat
- native voice chat with your Mac microphone and speakers
- browser and WebRTC voice chat

The default local language is Simplified Chinese.

English is supported, but it is a fallback path, not the default.

Blink is the assistant itself, not a generic helper wrapped around the product.
On the browser and native voice paths, Blink now keeps an explicit self-identity
layer, stable local session ids, and a local typed memory store across
sessions.

## Before You Start

The normal local stack is:

- Python `3.12`
- `uv`
- `ollama`
- Ollama model `qwen3.5:4b`

Install and bootstrap the local environment first:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
```

If you plan to use voice:

```bash
./scripts/bootstrap-blink-mac.sh --profile voice
```

If you plan to use browser voice:

```bash
./scripts/bootstrap-blink-mac.sh --profile browser
```

Then start Ollama:

```bash
ollama serve
```

## Choose The Right Mode

Use `text` if:

- you want the fastest and most reliable first run
- you are checking whether the assistant logic works
- you do not need microphone or speaker support yet

Use `voice` if:

- you want to talk directly through your Mac microphone and speakers
- you want the simplest spoken workflow without opening a browser

Use `browser` if:

- you want voice interaction in a browser UI
- you want WebRTC transport
- you want either primary browser path: `browser-zh-melo` for Chinese MeloTTS
  with camera, or `browser-en-kokoro` for English Kokoro with camera
- you want the main daily-use browser path, including the robot-head control
  surface when hardware embodiment is enabled
- you want the primary Blink brain path with persistent identity, memory, and
  bounded embodied actions
- you want camera/Moondream grounding with honest live camera state

## Quick Start

## 1. Terminal Chat

Start the text workflow:

```bash
./scripts/run-blink-chat.sh
```

This gives you the fastest baseline experience.

The assistant replies in Simplified Chinese by default.

For a hybrid OpenAI text demo, keep the same local shell but route only the LLM
layer remotely:

```bash
export OPENAI_API_KEY=...
./scripts/run-blink-chat-openai.sh
```

If you want English:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-chat.sh
```

## 2. Native Voice

Start the native voice workflow:

```bash
./scripts/run-blink-voice.sh
```

What to expect:

- it listens to your microphone
- it speaks through your system output
- the session stays in one speech language for the whole run

If you want English:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-voice.sh
```

For a hybrid OpenAI voice demo:

```bash
export OPENAI_API_KEY=...
./scripts/run-blink-voice-openai.sh
```

This still uses local speech recognition and local Kokoro TTS. Only text
generation uses OpenAI Responses.

## 3. Browser Voice

Start the browser workflow:

```bash
./scripts/run-blink-browser.sh
```

Then open:

```text
http://127.0.0.1:7860/client/
```

Allow microphone access in your browser when prompted.
If you are using local vision, also allow camera access.

If you want English:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-browser.sh
```

If you want the primary English-only browser path with Kokoro and no MeloTTS
sidecar:

```bash
./scripts/run-local-browser-kokoro-en.sh
```

This is the peer primary English browser/WebRTC conversation path. It keeps
MeloTTS off by default, keeps browser vision available, keeps protected playback on, and logs to
`artifacts/runtime_logs/latest-browser-kokoro-en.log`. It also ignores inherited
prompt overrides so an old Chinese `.env` prompt does not leak into this English
lane.

Blink now keeps stable product behavior in typed local runtime profiles under
`config/local_runtime_profiles.json`. Keep `.env` for local machine overrides
such as ports, device indexes, sidecar URLs, and secrets. Avoid putting product
language, prompt/persona behavior, camera policy, or barge-in defaults in
`.env` unless you are intentionally overriding a profile for debugging.

For the polished hybrid browser demo:

```bash
export OPENAI_API_KEY=...
./scripts/run-blink-browser-openai.sh
```

This keeps browser WebRTC, camera, local speech recognition, and local MeloTTS
on your machine. Only the LLM text-generation step is remote.

Before a demo, confirm the public runtime stack:

```bash
curl http://127.0.0.1:7860/api/runtime/stack
```

You should see the LLM provider/model, local STT/TTS choices, demo-mode state,
vision state, and bounded browser media state. The endpoint intentionally does
not expose keys, prompts, raw requests, browser device constraints, raw browser
errors, or private memory internals.

The browser page has a small model selector. It lists backend-curated local
Ollama Qwen choices by default. OpenAI Responses Mini/Nano style profiles are
locked unless the backend was launched through an OpenAI wrapper or the local
developer explicitly sets `BLINK_LOCAL_ENABLE_REMOTE_MODEL_SELECTION=1`. Pick a
model before connecting, or reconnect after changing it. The selector changes
only the LLM for the next browser session; speech language, STT, TTS, camera,
and persona behavior remain the ones selected when the backend was launched.

The browser page also includes the Blink operator workbench. Its Controls
section lets you inspect and update typed behavior controls, including humor,
vividness, sophistication, character presence, and story mode. The recommended
Witty Sophisticated preset is Chinese-first, vivid but compact, and
character-rich without pretending Blink is human.

The workbench Voice section also shows Input / STT health for the active
browser session. Use it to distinguish microphone/WebRTC problems from speech
recognition problems: microphone state reports whether audio frames are
arriving or stalled, while STT state reports whether Blink is idle, waiting for
a transcript, saw a transcript, or saw a bounded STT error. This surface does
not show raw transcript text. If the microphone is receiving frames but STT is
still waiting, the workbench shows a bounded STT wait age so you can separate
"mic is down" from "Whisper has not returned yet."

For richer local memory/personality setup, the Controls section includes a
Character/story seed preview. Preview is read-only. Applying it requires a
confirmation and an active local runtime store, then the workbench shows what
was written, skipped as no-op, or rejected. This does not delete existing
memory, change hidden prompts, switch models, or alter hardware policy.

### Actor Surface

The browser client shows Actor Surface v2 by default. It reads
`/api/runtime/actor-state` and `/api/runtime/actor-events`, then renders the
same components for Chinese/Melo and English/Kokoro:

- profile, language, TTS, camera/Moondream, current mode, and degradation state
- Heard
- Blink is saying
- Looking
- Used memory/persona
- Interruption
- a collapsed debug timeline of public-safe actor events

The surface may show bounded live text such as the current subtitle or bounded
heard text. It must not show secrets, raw SDP/ICE, raw images, raw audio,
hidden prompts, raw memory bodies, or unbounded transcript text.

### Browser Vision Behavior

With browser vision enabled, Blink keeps a public `camera_scene` state with
permission, track, frame freshness, on-demand Moondream status, and whether the
current answer actually used vision. This makes camera-on honest without
pretending continuous visual analysis is always happening.

By default, detailed vision is on-demand. When you explicitly ask what Blink can
see, it uses the bounded `fetch_user_image` tool to inspect the latest cached
camera frame with Moondream.

If the camera feed is stale or recovering, Blink should report `stale` or
`stalled` camera scene state instead of continuing to trust the last good frame.

Continuous low-frequency perception remains opt-in:

```bash
./scripts/run-local-browser.sh --vision --continuous-perception
```

Or slow it down:

```bash
./scripts/run-local-browser.sh --vision --continuous-perception --continuous-perception-interval-secs 5
```

If you want to validate the upgraded memory, commitments, reflection, audit,
and browser presence stack end to end, use the checked-in
[Continuity Test Manual](./CONTINUITY_TEST_MANUAL.md).

## Native English Voice With Camera

The default native English voice path remains camera-free:

```bash
./scripts/run-local-voice-en.sh
```

For English-only voice with camera grounding outside the browser, use the
macOS helper path:

```bash
./scripts/bootstrap-blink-mac.sh --profile voice --with-vision
./scripts/build-macos-camera-helper.sh
./scripts/run-local-voice-macos-camera-en.sh
```

This path uses Mac mic, MLX Whisper, Ollama, Kokoro, and Mac speaker output.
`BlinkCameraHelper.app` owns camera permission and writes a low-FPS local
snapshot cache. Blink reads the latest fresh RGB snapshot only when
`fetch_user_image` is invoked, then uses Moondream to summarize the image for
the LLM.

This path does not start browser/WebRTC, MeloTTS, local-http-wav, or the retired
OpenCV native camera capture path. It is for English Kokoro testing only.
The helper contract is documented in
[Native macOS Camera Helper](./native-macos-camera-helper.md).

## How Chinese Voice Works

The current Chinese local product path has two levels:

1. Native English voice path
   This uses Kokoro.
   It is easy to run and useful for local iteration.

2. Higher-quality Chinese browser path
   This uses MeloTTS behind `local-http-wav`.
   It is the recommended path when you want stronger Mandarin voice quality.

Important:

- Chinese browser sessions prefer Melo automatically when the sidecar is running
  and you did not pin another backend.
- Kokoro remains the bootstrap and fallback path because it is easy to run
  locally.
- MeloTTS is the recommended quality path for Chinese speech.

## Recommended Chinese Voice Upgrade

If spoken Chinese is working but you want a better Mandarin voice, use the Melo
sidecar.

Bootstrap the Melo environment:

```bash
./scripts/bootstrap-melotts-reference.sh
```

Run the primary Chinese browser voice path already pointed at Melo:

```bash
./scripts/run-local-browser-melo.sh
```

That browser Melo wrapper now keeps camera support on by default. Open the
normal Blink UI at `http://127.0.0.1:7860/client/`, allow both microphone and
camera access, and reload the page once if it was already open before the
launcher restarted.

If you start the Melo server yourself, it is an API service, not the browser UI.

Typical API endpoints:

```text
http://127.0.0.1:8001/healthz
http://127.0.0.1:8001/voices
```

The browser UI is still:

```text
http://127.0.0.1:7860/client/
```

That `/client` page is the repo-owned Blink browser UI.
This repository owns both the local Blink backend behavior and the browser
branding surface mounted at that route.

## Native English Voice

For the stable non-browser voice loop, use the native English path:

```bash
./scripts/bootstrap-blink-mac.sh --profile voice
# or, for an already bootstrapped checkout:
uv sync --python 3.12 --group dev \
  --extra local \
  --extra kokoro \
  --extra mlx-whisper

ollama serve
./scripts/run-local-voice-en.sh
```

This mode uses the Mac microphone, MLX Whisper, the configured local LLM, and
Kokoro. It is English-only and intentionally does not use camera, Moondream,
browser/WebRTC, or MeloTTS. Use the browser path when camera grounding is
needed. Logs are written to `artifacts/runtime_logs/latest-native-voice-en.log`.

## Audio Behavior You Should Understand

### Speech language is fixed per session

If you start a Chinese session, the spoken voice path stays Chinese for that
session.

Telling the assistant "speak English now" may change text generation, but it
does not hot-swap the TTS engine during the run.

If you want spoken English, start a fresh English session:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-voice.sh
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-browser.sh
```

### Native voice interruption

On laptop speakers, the assistant can accidentally interrupt itself if its own
voice leaks back into the microphone.

Because of that, the lower-level native local voice command defaults to
protected playback:

```bash
./scripts/run-blink-voice.sh
```

The English wrapper commands default to protected playback on native PyAudio, so
Blink can finish answers on open speakers:

```bash
./scripts/run-local-voice-en.sh
./scripts/run-local-voice-macos-camera-en.sh
```

For live voice barge-in, use headphones or another echo-safe setup and opt in:

```bash
./scripts/run-local-voice-en.sh --allow-barge-in
./scripts/run-local-voice-macos-camera-en.sh --allow-barge-in
```

For the lower-level command, opt into barge-in explicitly:

```bash
./scripts/run-blink-voice.sh --allow-barge-in
```

Or:

```bash
BLINK_LOCAL_ALLOW_BARGE_IN=1 ./scripts/run-blink-voice.sh
```

The browser/WebRTC voice path also defaults to protected playback. This is
server-side turn muting while Blink is speaking, not a browser media-track
mutation: Blink does not patch `getUserMedia`, disable the microphone/camera
tracks, or use WebRTC renegotiation for this protection. `--allow-barge-in`
disables the protection when you explicitly want to test browser interruption on
an echo-safe setup. If the browser ever cuts off later sentences again, check
`/api/runtime/voice-metrics` for interruption and flush counts before blaming
MeloTTS.

## Optional Camera Support In Browser Mode

`./scripts/run-blink-browser.sh` now starts in Chinese with camera support on by
default, and it prefers the MeloTTS `local-http-wav` path when you do not pin a
different backend. You can still override that behavior explicitly:

```bash
./scripts/bootstrap-blink-mac.sh --profile browser --with-vision
./scripts/run-blink-browser.sh
./scripts/run-blink-browser.sh --robot-head-driver live --robot-head-port /dev/cu.usbmodemXXXX --robot-head-live-arm
./scripts/run-blink-browser.sh --tts-backend kokoro
./scripts/run-local-browser-melo.sh
./scripts/run-local-browser.sh --vision --tts-backend kokoro
```

Camera support still requires the optional vision stack, but the canonical
Blink browser launchers now assume you want that mode by default.

After a fresh bootstrap or `.venv` rebuild, the browser path may need around
10-20 seconds before the page is actually ready, especially when vision is on.
The launcher now waits for real `/client/` readiness and only prints the URL after the page is
serving at `http://127.0.0.1:7860/client/`.

When you start browser voice through `./scripts/run-blink-browser.sh`, that
launcher now owns the Blink browser runtime and any sidecar it started itself.
Pressing `Ctrl+C` in that terminal should stop both cleanly. If you pointed
Blink at an already-running sidecar, Blink leaves that existing sidecar alone.

If the browser sits idle after you connect, Blink now rate-limits repeated
WebRTC timeout warnings. You should see the first real mic or camera stall, but
the terminal should not flood with the same warning every two seconds.

What to expect in camera mode:

- you still open the normal browser UI at `http://127.0.0.1:7860/client/`
- the browser must be allowed to use the camera and microphone
- if Blink never receives a real camera track, camera questions fail honestly
  until you reload/reconnect with working camera permission and device access
- the assistant does not constantly describe the video stream
- it looks at the latest camera frame only when you ask a camera question

Good test prompts:

- `请描述一下你现在看到的画面`
- `我手里拿着什么？`
- `我身后有什么？`

If you also enabled the robot head in the browser runtime, good head-control
prompts are:

- `眨眼一次`
- `向左看一下`
- `向右看一下`
- `回到中位`
- `现在头部状态是什么？`
- `请读一下屏幕上的大字`

The browser and native voice paths now also support bounded memory operations.
These are the stable ways to tell Blink to remember or update something:

- `记住我叫阿周。`
- `记住我喜欢机器人。`
- `记住提醒我晚上给妈妈打电话。`
- `更正一下，我不喜欢咖啡。`
- `忘记我的名字。`
- `把“给妈妈打电话”这件事标记为完成。`

Under the hood, Blink now stores this in a local SQLite brain database at
`~/.cache/blink/brain/brain.db`. That brain keeps:

- pinned Blink identity and policy blocks
- per-user facts and preferences
- tasks and reminders
- recent turn episodes and summaries
- robot and vision presence state

The default browser path is intentionally constrained. Blink should not invent
arbitrary robot motion from open-ended language. Normal daily use only exposes a
finite, safe action set such as blink, look left, look right, return neutral,
and status. Lower-level robot-state tools exist only in explicit operator mode.

## Common Commands

Run a one-shot text test:

```bash
uv run --python 3.12 blink-local-chat --once "Explain Blink in one sentence."
```

List audio devices:

```bash
./scripts/run-blink-voice.sh --list-audio-devices
```

Pick specific audio devices:

```bash
./scripts/run-blink-voice.sh --input-device 1 --output-device 3
```

Run doctor checks:

```bash
./scripts/doctor-blink-mac.sh --profile text
./scripts/doctor-blink-mac.sh --profile voice
./scripts/doctor-blink-mac.sh --profile browser
./scripts/doctor-blink-mac.sh --profile browser --llm-provider openai-responses --demo-mode
```

Generate comparison WAV files:

```bash
./scripts/eval-local-tts.sh
./scripts/eval-local-tts.sh --backend kokoro --all-kokoro-zh-voices
./scripts/eval-local-tts.sh --backend local-http-wav --language zh
```

## Troubleshooting

### The browser page opens, but the AI does not respond

Check these in order:

1. Make sure `ollama serve` is running.
2. Make sure you opened `http://127.0.0.1:7860/client/`.
3. Make sure the browser has microphone permission.
4. Open `/api/runtime/stack` and check `browser_media`. A disconnected,
   unavailable, stalled, or stale camera state means real browser media did not
   reach Blink or stopped reaching Blink.
5. If you are using Melo, confirm the sidecar is healthy:

```bash
curl http://127.0.0.1:8001/healthz
```

6. Run the doctor command:

```bash
./scripts/doctor-blink-mac.sh --profile browser
```

If you are validating the browser workflow itself, there is now an opt-in local
Playwright smoke test:

```bash
uv run python -m playwright install chromium
BLINK_RUN_BROWSER_E2E=1 uv run pytest -m browser_e2e
```

### I saw a microphone or camera stall warning in the terminal

One warning by itself does not necessarily mean the session is broken.

Blink now rate-limits repeated SmallWebRTC stall warnings, so the terminal
should show the first real microphone or camera stall instead of flooding every
two seconds.

If the session recovered on its own, that one warning is expected current
noise. If audio or camera really stopped working, refresh the page once and
retry before assuming the backend is down.

SmallWebRTC reports backend receive-track stalls, but the transport no longer
asks for immediate peer renegotiation from the stall callback. If the page still
shows local camera/microphone preview but Blink stops receiving audio or video
frames, refresh or reconnect the `/client/` page before changing TTS or model
settings.

The local browser client intentionally does not auto-renegotiate or auto-restart
ICE after a successful first connection. If the first answer works but the next
question gets no response, reload the browser page once and reconnect before
debugging Ollama, MLX Whisper, or MeloTTS.

For the browser camera path specifically, Blink now also:

- marks visual presence as uncertain when the camera feed is stale or stalled
- rejects stale cached frames in `fetch_user_image`
- reports `camera_manual_reload_required` instead of automatically mutating the
  active WebRTC session
- exposes the latest visual-health state through `blink-local-doctor` and
  `blink-local-brain-audit`

### The browser camera is on, but the answer is still vague

This is different from a camera-permission failure.

If the browser page is connected and the backend is receiving a video track, the
remaining weak point is usually the local vision path, not TTS.

Typical causes:

1. the visible text is too small or blurry in the camera frame
2. the scene does not have enough lighting or contrast
3. the local vision model can describe the scene only roughly

What to try:

1. hold the object or screen closer to the camera
2. ask about large objects or large text first
3. keep the scene steady for a moment before asking
4. retry with a more direct question such as `我手里拿着什么？`

### The voice sounds too accented in Chinese

That usually means you are still on the bootstrap Kokoro path.

Move to the recommended browser Melo path:

```bash
./scripts/bootstrap-melotts-reference.sh
./scripts/run-local-browser-melo.sh
```

### The assistant stops speaking too early in native voice mode

The native path already protects playback by default.

If you enabled barge-in manually, disable it and retry:

```bash
./scripts/run-blink-voice.sh
```

If the problem persists, run:

```bash
./scripts/doctor-blink-mac.sh --profile voice
```

### The first browser answer works, but the second sounds wrong

This repeated failure has a dedicated debugging note in
[debugging.md](./debugging.md).

Do not change the model, STT, camera permission path, or WebRTC recovery first.
Check the separated health surfaces:

```bash
curl http://127.0.0.1:7860/api/runtime/voice-metrics
curl http://127.0.0.1:7860/api/runtime/stack
tail -n 200 artifacts/runtime_logs/latest-browser-melo.log
```

The usual causes are camera frame-rate pressure, too many tiny MeloTTS WAV
chunks, an overlarge Melo chunk, or a local HTTP WAV timeout. The expected path
uses moderate bounded `local-http-wav` chunks and explicit browser
reload/reconnect if backend media frames stop.

### I opened the Melo port in a browser and saw no UI

That is expected.

The Melo sidecar is an API service, not the product UI.

Use:

- `http://127.0.0.1:8001/healthz` for the Melo API health check
- `http://127.0.0.1:7860/client/` for the browser voice UI

You may still see some third-party Melo reference-env warnings in the terminal
when it starts. Those are known dependency warnings, not a sign that you opened
the wrong port.

## Recommended First-Time Workflow

If you are new to the product, use this order:

1. Run terminal chat first.
2. Move to native voice if text works.
3. Move to browser voice if you want a browser interface.
4. Move from Kokoro to MeloTTS if Chinese speech quality matters.

That order gives you the fastest path to a working system with the smallest
number of moving parts.

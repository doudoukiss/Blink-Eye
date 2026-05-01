# Local Development

This repository now has a practical local-first workflow for a MacBook Pro on Apple Silicon. Blink is the canonical product identity for that workflow. The runtime installs through the `blink-ai` package and the `blink` import namespace.

If you want a user-facing guide instead of the engineering workflow notes, start
with [`docs/USER_MANUAL.md`](./docs/USER_MANUAL.md).

The default local interaction mode is now Simplified Chinese. English remains a supported fallback, but not the default.

Canonical local surface:

- canonical local CLI family: `blink-local-*`
- canonical local wrapper scripts: `./scripts/*blink*`
- canonical local env prefix: `BLINK_*`

## Profiles

Use the bootstrap profile that matches the workflow you need:

```bash
./scripts/bootstrap-blink-mac.sh --profile text
./scripts/bootstrap-blink-mac.sh --profile voice
./scripts/bootstrap-blink-mac.sh --profile browser
./scripts/bootstrap-blink-mac.sh --profile full
```

If you want the repo to download the selected local models before first use,
add `--prefetch-assets`:

```bash
./scripts/bootstrap-blink-mac.sh --profile browser --with-vision --prefetch-assets
```

Profile behavior:

- `text`: terminal chat only
- `voice`: native mic/speaker voice loop with local STT and Kokoro
- `browser`: local browser/WebRTC voice loop with the same local STT and TTS stack
- `full`: native voice plus browser/WebRTC together
- `--with-vision`: adds the optional local Moondream vision stack for browser
  camera inspection or the separate macOS helper voice path

The browser UI mounted at `http://127.0.0.1:7860/client/` is now a repo-owned
Blink browser bundle. This repository owns the `/client` route, branding,
runtime policy, and backend integration behind that path.

This public checkout is source-first: disposable build output is absent, while
required runtime assets are retained when Blink loads them directly. The
browser client workspace lives in
[`web/client_src/src`](./web/client_src/src); it contains authored Blink
overlays plus the vendored browser runtime assets needed for `/client/`. If a
local package build needs generated browser copies, rebuild them with:

```bash
node web/client_src/build.mjs
```

The generated `src/blink/web/client_dist/` directory is intentionally ignored.

Browser/WebRTC is the daily-use brain surface. Its two primary profiles are
`browser-zh-melo` and `browser-en-kokoro`; treat both as equal main
browser/WebRTC product lanes. They share the `BrainRuntime` surface for
identity, typed memory, bounded explicit memory tools, bounded robot-head
actions, and browser-first presence handling. The native voice path reuses that
same runtime for backend-isolation work instead of maintaining a separate local
brain layer.

The bootstrap script creates `.env` from [`env.local.example`](./env.local.example) if `.env` does not exist yet.

You can also prefetch after bootstrap without reinstalling anything:

```bash
./scripts/prefetch-blink-assets.sh --profile text
./scripts/prefetch-blink-assets.sh --profile voice
./scripts/prefetch-blink-assets.sh --profile browser --with-vision
./scripts/prefetch-blink-assets.sh --profile voice --with-vision
./scripts/prefetch-blink-assets.sh --profile voice --tts-backend piper
```

## Recommended Local Stack

- Python `3.12`
- `uv`
- `ollama`
- Ollama model `qwen3.5:4b`
- MLX Whisper
- Kokoro for zero-setup bootstrap
- `browser-zh-melo`: MeloTTS via `local-http-wav` for the Chinese browser path
- `browser-en-kokoro`: English browser Kokoro via
  `./scripts/run-local-browser-kokoro-en.sh`
- Smart Turn v3

Fallbacks supported by the new local CLIs:

- STT: `whisper` instead of `mlx-whisper`
- TTS: `xtts`, `local-http-wav`, or `piper`

## Chinese-Default Local Stack

The default local product path is:

- language: `zh`
- STT: `mlx-whisper` with `mlx-community/whisper-medium-mlx`
- LLM: Ollama `qwen3.5:4b`
- TTS: `kokoro`
- text chat default: concise developer-focused Chinese
- voice/browser default: shorter speech-safe Chinese for TTS

Why this changed:

- Mandarin quality is primarily a TTS problem, not an Ollama/Qwen problem.
- The current Ollama adapter ignores the `developer` role, so long-lived Chinese response constraints must live in `system_instruction`.
- The default MBP path now stays on an in-repo backend so the local setup actually runs on Apple Silicon without extra infrastructure.
- The recommended Chinese-quality path is now MeloTTS behind `local-http-wav`.
- Chinese voice and browser runtime now auto-prefer a healthy `local-http-wav`
  sidecar when no backend is explicitly pinned.
- XTTS and CosyVoice remain optional secondary paths behind the same seam.

The main design references for this workflow live in [`docs`](./docs):

- [`docs/chinese-conversation-adaptation.md`](./docs/chinese-conversation-adaptation.md)
- [`docs/roadmap.md`](./docs/roadmap.md)
- [`docs/USER_MANUAL.md`](./docs/USER_MANUAL.md)

Supplementary engineering references:
- [`CHINESE_TTS_UPGRADE.md`](./CHINESE_TTS_UPGRADE.md)
- [`LOCAL_CAPABILITY_MATRIX.md`](./LOCAL_CAPABILITY_MATRIX.md)

## Default Commands

### Terminal chat

```bash
./scripts/bootstrap-blink-mac.sh --profile text
ollama serve
./scripts/run-blink-chat.sh
```

The text path now replies in Simplified Chinese by default.

English fallback:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-chat.sh
```

One-shot smoke test:

```bash
uv run --python 3.12 blink-local-chat --once "Explain Blink in one sentence."
```

Brain-backed text validation:

```bash
uv run --python 3.12 blink-local-brain-chat --once "用一句话解释 Blink。"
uv run --python 3.12 blink-local-brain-audit --reply-query "Blink 现在记得我什么？"
```

Hybrid OpenAI demo lane:

```bash
export OPENAI_API_KEY=...
./scripts/run-blink-chat-openai.sh --once "用一句话解释 Blink。"
```

This keeps the local chat wrapper narrow: only the LLM provider changes to
OpenAI Responses, and demo mode is enabled by default for bounded output.

Recommended validation ladder:

1. `blink-local-chat` for a simple Ollama sanity check.
2. `blink-local-brain-chat` for continuity graph, dossier, and active-context validation.
3. `blink-local-brain-audit` for operator-facing graph/governance/context-packet inspection.

### Native voice

```bash
./scripts/bootstrap-blink-mac.sh --profile voice
ollama serve
./scripts/run-blink-voice.sh
```

Native voice does not use MeloTTS. For Mandarin speech quality and camera
testing, use the browser Melo path.

Hybrid OpenAI voice demo lane:

```bash
export OPENAI_API_KEY=...
./scripts/run-blink-voice-openai.sh
```

This path keeps STT local, keeps Kokoro TTS local, and sends only the LLM
text-generation step to OpenAI Responses. The wrapper defaults to
`gpt-5.4-mini` for the low-latency demo lane. Override with
`BLINK_LOCAL_OPENAI_RESPONSES_MODEL=gpt-5.4` when you want a hero-flow quality
check.

The system-level record of why this stack is structured this way is
[`docs/chinese-conversation-adaptation.md`](./docs/chinese-conversation-adaptation.md).

For Chinese `run-local-voice` and `run-local-browser`, the runtime policy is:

- explicit `BLINK_LOCAL_TTS_BACKEND` or `--tts-backend` always wins
- otherwise, if a healthy `local-http-wav` sidecar is reachable, Blink uses it
- if no sidecar is reachable, Blink falls back to Kokoro

If you want to point at an already running local HTTP WAV service directly:

```bash
BLINK_LOCAL_TTS_BACKEND=local-http-wav \
BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001 \
./scripts/run-blink-voice.sh

BLINK_LOCAL_TTS_BACKEND=xtts \
BLINK_LOCAL_XTTS_BASE_URL=http://127.0.0.1:8000 \
./scripts/run-blink-voice.sh
```

Native voice is now the English-only Kokoro path:

```bash
./scripts/run-local-voice-en.sh
```

### Native English Voice

Use this path when you want the lowest-risk non-browser voice loop. It keeps the
conversation English-only and uses the local pipeline:

```text
Mac mic -> MLX Whisper -> LLM -> Kokoro
```

Install the native voice extras:

```bash
./scripts/bootstrap-blink-mac.sh --profile voice
# or, for an already bootstrapped checkout:
uv sync --python 3.12 --group dev \
  --extra local \
  --extra kokoro \
  --extra mlx-whisper
```

Run it:

```bash
ollama serve
./scripts/run-local-voice-en.sh
```

Native voice does not use the camera, Moondream, browser/WebRTC, or MeloTTS.
Camera grounding stays in the browser path. The old
`./scripts/run-local-voice-camera-en.sh` wrapper is kept only as a compatibility
alias and starts the same camera-free English voice path.

The wrapper logs to `artifacts/runtime_logs/latest-native-voice-en.log`.

### Native English Voice With macOS Camera Helper

Use this path only when you want English Kokoro voice plus camera grounding
without browser/WebRTC. It keeps the voice loop English-only:

```text
Mac mic -> MLX Whisper -> LLM -> Kokoro -> Mac speaker
BlinkCameraHelper.app -> latest RGB snapshot -> Moondream on demand
```

This path uses a minimal native macOS helper app because camera permission is
owned by the app that opens the camera. It does not use Terminal/OpenCV camera
capture, MeloTTS, browser/WebRTC, or continuous video ingestion. Moondream runs
only when the `fetch_user_image` tool asks for the latest helper snapshot.

Build the helper and run the path:

```bash
./scripts/bootstrap-blink-mac.sh --profile voice --with-vision
./scripts/build-macos-camera-helper.sh
./scripts/run-local-voice-macos-camera-en.sh
```

On the first run, macOS should prompt for camera access for
`BlinkCameraHelper`. If permission is denied, allow `BlinkCameraHelper` in
System Settings > Privacy & Security > Camera, then restart the wrapper.

The wrapper logs to
`artifacts/runtime_logs/latest-native-voice-macos-camera-en.log`. Helper state
for each run is written under `artifacts/runtime_logs/blink-camera-helper-*`
with `status.json` and `latest.rgb`.

See [`docs/native-macos-camera-helper.md`](./docs/native-macos-camera-helper.md)
for the helper status contract and debugging boundary.

List audio devices:

```bash
./scripts/run-blink-voice.sh --list-audio-devices
```

Choose explicit devices:

```bash
./scripts/run-blink-voice.sh --input-device 1 --output-device 3
```

On MacBook Pro setups with an external display attached, the native voice flow
will auto-prefer the built-in MacBook microphone and speakers when macOS has
routed the system defaults through display audio. Use `--input-device` and
`--output-device` to override that behavior explicitly.

Lower-level native local audio defaults to protected playback: while the
assistant is speaking, Blink mutes user-turn detection so speaker bleed does not
cut the response off after a second or two. If you use the lower-level launcher
and want live barge-in on a headset or another echo-safe setup, opt in
explicitly:

```bash
./scripts/run-blink-voice.sh --allow-barge-in
```

Or set:

```bash
BLINK_LOCAL_ALLOW_BARGE_IN=1
```

The English native wrappers also default to protected playback when using native
PyAudio. This is deliberate: unlike the browser path, PyAudio does not provide
WebRTC echo cancellation, so built-in speakers can leak Blink's own voice back
into the MacBook microphone and make it interrupt itself after a word or two.

```bash
./scripts/run-local-voice-en.sh
./scripts/run-local-voice-macos-camera-en.sh
```

Those wrappers force `BLINK_LOCAL_ALLOW_BARGE_IN=0`, so an inherited shell or
`.env` value cannot silently put the English path into self-interrupting
barge-in mode.

For live voice barge-in, use headphones or another echo-safe setup and opt in
explicitly:

```bash
./scripts/run-local-voice-en.sh --allow-barge-in
./scripts/run-local-voice-macos-camera-en.sh --allow-barge-in
```

When barge-in is enabled, the interruption path now also tells the native
PyAudio output transport to abort the active playback write and restart the
stream. If the startup line says `barge_in=on` but speech still cannot be
interrupted, debug the local output device/driver path before changing STT,
LLM, TTS, or camera code.

If a native wrapper self-interrupts or cuts off speech early, return to protected
playback by running it without `--allow-barge-in`.

Browser/WebRTC voice now also defaults to protected playback, but the protection
is server-side turn muting only. Blink suppresses user-turn detection while the
bot is speaking so speaker bleed does not interrupt TTS, but it does not patch
`getUserMedia`, disable browser microphone/camera tracks, or trigger WebRTC
renegotiation. `--allow-barge-in` disables this browser protection for explicit
barge-in tests on echo-safe setups.

### Browser/WebRTC voice

```bash
./scripts/bootstrap-blink-mac.sh --profile browser
ollama serve
./scripts/run-blink-browser.sh
```

Then open:

```text
http://127.0.0.1:7860/client/
```

If you are using the recommended Melo sidecar, remember that the sidecar port is
the TTS API, not the browser UI. Typical checks are:

```text
http://127.0.0.1:8001/healthz
http://127.0.0.1:8001/voices
```

English fallback:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-browser.sh
```

Primary English browser Kokoro with browser vision:

```bash
./scripts/run-local-browser-kokoro-en.sh
```

This is the peer primary English browser/WebRTC path when you want Kokoro
without MeloTTS or continuous perception load. It keeps browser vision available
by default, keeps continuous perception off by default, writes to
`artifacts/runtime_logs/latest-browser-kokoro-en.log`, and ignores inherited
prompt overrides so an old Chinese `.env` prompt does not make the English lane
reply in Chinese.

The two primary browser wrappers are `./scripts/run-local-browser-melo.sh` for
`browser-zh-melo` and `./scripts/run-local-browser-kokoro-en.sh` for
`browser-en-kokoro`. The lower-level browser command still supports manual
overrides:

```bash
./scripts/bootstrap-blink-mac.sh --profile browser --with-vision
ollama serve
./scripts/run-blink-browser.sh
./scripts/run-blink-browser.sh --robot-head-driver live --robot-head-port /dev/cu.usbmodemXXXX --robot-head-live-arm
./scripts/run-blink-browser.sh --tts-backend kokoro
./scripts/run-local-browser.sh --vision --tts-backend kokoro
```

The browser actor runtime is the main product surface for both primary
profiles. The browser client reads `/api/runtime/actor-state` and
`/api/runtime/actor-events` to show profile/language/TTS, listening, thinking,
looking, speaking, camera scene, interruption, memory/persona, and degradation
state. The v1 `/api/runtime/performance-state` and
`/api/runtime/performance-events` endpoints remain available for compatibility.

Actor traces are off by default. Enable sanitized JSONL traces only for focused
debugging:

```bash
BLINK_LOCAL_ACTOR_TRACE=1 ./scripts/run-local-browser-melo.sh
BLINK_LOCAL_ACTOR_TRACE=1 ./scripts/run-local-browser-kokoro-en.sh
```

Hybrid OpenAI browser/WebRTC demo lane:

```bash
export OPENAI_API_KEY=...
./scripts/run-blink-browser-openai.sh
```

This is the recommended investor-demo wrapper: browser camera and WebRTC stay
local, STT stays local, MeloTTS stays local, and only the LLM layer uses
OpenAI Responses. Confirm the public runtime stack before a demo with:

```bash
curl http://127.0.0.1:7860/api/runtime/stack
```

The stack payload intentionally shows provider/model/STT/TTS/demo/vision state
plus bounded browser media state. It does not expose API keys, prompts, request
bodies, raw device constraints, browser exception text, or internal memory
provenance.

The browser UI also includes a compact model selector backed by the safe
`/api/runtime/models` catalog. It exposes curated local Qwen profiles through
Ollama by default. Remote OpenAI Responses profiles such as Nano, Mini, and the
quality tier stay locked in the selector unless you explicitly launch an
OpenAI-backed wrapper or set `BLINK_LOCAL_ENABLE_REMOTE_MODEL_SELECTION=1` for a
demo/development run. A change applies to the next browser session or reconnect;
it does not hot-swap a model inside an active WebRTC conversation. Speech
language is still fixed when the session starts, so use `BLINK_LOCAL_LANGUAGE=en`
before launching if you want the English speech path.

The current browser camera path works like this:

- SmallWebRTC owns microphone/camera acquisition and keeps the browser camera
  track flowing
- the extra browser startup asset is passive only: it retries playback for
  autoplay media elements, but it does not wrap `getUserMedia`, stop tracks,
  retry audio-only capture, or mutate permissions/device constraints
- `/api/runtime/stack` reports the bounded backend-observed media state and any
  explicitly posted public client-media state without exposing raw browser
  errors or device constraints
- Blink caches the latest camera frame locally
- the browser runtime exposes public camera scene state for permission, track
  health, latest frame sequence/age, freshness, on-demand vision status, and
  whether the current answer actually used vision
- when the user asks about the camera view, the assistant calls `fetch_user_image`
- the browser runtime runs local Moondream analysis on that cached frame and
  returns the result to the assistant as tool output
- `fetch_user_image` now rejects stale cached frames instead of quietly
  reusing old video

Continuous perception is browser-first, opt-in, and intentionally lightweight:

- it is disabled by default for both `browser-zh-melo` and `browser-en-kokoro`
- it can be enabled with `--continuous-perception` or
  `BLINK_LOCAL_CONTINUOUS_PERCEPTION=1`
- it does not replace `fetch_user_image`
- it does not push every camera frame into the model
- stale or recovering camera feeds now degrade to explicit `stale` or `stalled`
  camera scene states; they do not keep the last good `available` state
  indefinitely

Useful browser overrides:

```bash
./scripts/run-local-browser.sh --vision --continuous-perception
./scripts/run-local-browser.sh --vision --continuous-perception --continuous-perception-interval-secs 5
./scripts/run-local-browser.sh --vision --robot-head-driver simulation
```

For a practical end-to-end validation workflow of memory, reflection,
commitments, audits, and browser presence, use the checked-in
[Continuity Test Manual](./docs/CONTINUITY_TEST_MANUAL.md).

This means a camera answer can fail even when the browser has camera permission:
poor answers may come from low-detail frames, weak local vision prompts, or
Moondream limitations rather than from TTS or WebRTC bootstrap. Browser
presence failures are now more often transport-health or stale-frame problems
than raw VLM-classification problems.

The browser and native voice runtimes now also share the same bounded local
brain surface:

- stable Blink identity blocks loaded from `src/blink/brain/defaults/`
- local SQLite brain store at `~/.cache/blink/brain/brain.db`
- stable local brain user id `local_primary` by default, with
  `BLINK_LOCAL_BRAIN_USER_ID` available when you intentionally want a named
  local profile
- typed memory for profile facts, preferences, tasks, episodes, presence, and
  embodied action events
- bounded `brain_*` tools for remember, forget, and task completion flows
- bounded `robot_head_*` daily-use actions rather than raw state or motif tools

The raw `robot_head_set_state` and `robot_head_run_motif` tools are no longer
part of the default daily browser path. They are only exposed when you start
the runtime in explicit robot-head operator mode.

After a fresh bootstrap or `.venv` rebuild, the browser launcher can take
around 10-20 seconds to finish local vision startup before `http://127.0.0.1:7860/client/`
is actually ready. The launcher now waits for real `/client/` HTTP readiness and only prints the
ready URL after the page is serving.

Shutdown behavior is now explicit too:

- `./scripts/run-blink-browser.sh` keeps ownership of the browser runtime and
  any Melo/CosyVoice sidecar that it started itself
- pressing `Ctrl+C` in that launcher tears down the browser runtime and the
  sidecar it spawned
- if the launcher attached to an already-running sidecar, it leaves that
  pre-existing sidecar alone

Idle browser/WebRTC stalls are now warning-throttled. You should still see the
first microphone or camera timeout if media stops flowing, but Blink no longer
spams the terminal every two seconds while the same idle condition persists.
SmallWebRTC track-stall callbacks report the structured stall state but no
longer trigger immediate peer renegotiation themselves. If the browser preview
is still visible but Blink stops receiving microphone or camera frames, refresh
or reconnect the `/client/` page first. The default browser camera health manager
is diagnostic-only: it records stale/stalled state and `camera_manual_reload_required`
instead of mutating the active WebRTC session.

The packaged local browser client also avoids automatic ICE reconnect and
server-triggered renegotiation in the daily Melo + camera path. If media stops
after a successful first turn, treat reload/reconnect as the supported recovery
path before changing model, STT, or TTS settings.

Current residual warning noise to recognize, not overreact to:

- one-time SmallWebRTC microphone or camera stall warnings can still appear
  around reconnects or browser media hiccups
- the repo-owned Melo reference env still emits some third-party deprecation
  and future warnings on startup (`pkg_resources`, `weight_norm`,
  `resume_download`)
- these are known debt, not the main functional failure surface

### Diagnostics

```bash
./scripts/doctor-blink-mac.sh --profile text
./scripts/doctor-blink-mac.sh --profile voice
./scripts/doctor-blink-mac.sh --profile browser
./scripts/doctor-blink-mac.sh --profile full --with-vision
./scripts/doctor-blink-mac.sh --profile browser --llm-provider openai-responses --demo-mode
```

The doctor command checks:

- `uv`
- `ollama`
- `brew` and `portaudio` when native voice is requested
- Python extras needed by the selected profile
- Ollama reachability
- Ollama model availability
- XTTS or local HTTP WAV server reachability when a voice/browser profile uses those backends
- audio-device enumeration when native voice is installed
- deterministic local model caches for STT, Smart Turn, and optional vision
- packaged deterministic browser presence-detector readiness
- optional VLM enrichment readiness
- the latest stored browser visual-health state if a browser runtime already wrote one
- OpenAI Responses key/model/tier/demo-mode configuration when
  `--llm-provider openai-responses` is selected

For visual debugging, the most useful operator commands are now:

```bash
uv run --python 3.12 blink-local-doctor --profile browser --with-vision
uv run --python 3.12 blink-local-brain-audit --runtime-kind browser
```

The audit output now includes a dedicated visual-health section with camera
track state, last fresh frame time, frame age, detector backend/confidence,
recovery attempts, and the current degraded reason.

## Session Speech Semantics

The local voice and browser flows choose one TTS service when the session starts.
That service stays fixed for the session.

That means:

- a Chinese session uses the Chinese voice path selected at startup
- an English session uses the English voice path selected at startup
- telling the assistant "speak English now" inside a Chinese session does not hot-swap the TTS engine

If you want spoken English, start an English session explicitly:

```bash
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-voice.sh
BLINK_LOCAL_LANGUAGE=en ./scripts/run-blink-browser.sh
```

This routing rule is deliberate. The local product path does not use runtime `TTSUpdateSettingsFrame` switching.

## Configuration Profiles

Stable product paths now use typed local runtime profiles from
[`config/local_runtime_profiles.json`](./config/local_runtime_profiles.json).
The current checked-in profiles are:

- `browser-zh-melo`: primary Chinese browser/WebRTC path, MeloTTS through
  `local-http-wav`, camera on, continuous perception off.
- `browser-en-kokoro`: primary English browser/WebRTC path, Kokoro, camera on,
  protected playback, inherited prompt overrides ignored.
- `native-en-kokoro`: backend-isolation English native PyAudio path, Kokoro,
  camera off, protected playback.
- `native-en-kokoro-macos-camera`: separate English native helper-camera path,
  Kokoro, macOS camera helper, Moondream only on explicit image-tool use.

Native English Kokoro is for backend isolation, not daily camera or
interruption UX. See
[`docs/debugging/native_voice_isolation.md`](./docs/debugging/native_voice_isolation.md)
for the expected `runtime=native transport=PyAudio ... barge_in=off` status
lines and the protected-playback policy.

Use `.env` for machine facts and secrets: ports, device indexes, local paths,
API keys, and sidecar URLs. Avoid storing product behavior, language contracts,
prompt/persona text, or media policy in `.env`.

Resolution order for browser and native voice is:

1. code defaults
2. checked-in typed profile
3. optional gitignored local profile override at `.blink-local-profiles.json` or
   `BLINK_LOCAL_CONFIG_PROFILES_PATH`
4. `.env` / exported environment variables
5. CLI flags

The wrappers set the right profile and also pin the high-risk path settings
explicitly, so an old `.env` cannot silently turn the English Kokoro browser lane
back into a Chinese or camera-heavy session.

## Environment Surface

The local workflows still read these values from `.env` for local machine
overrides:

```dotenv
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=qwen3.5:4b
# Optional shared override. If set, this replaces both the default text-chat
# prompt and the default speech prompt.
# OLLAMA_SYSTEM_PROMPT=你正在帮助开发者在本地探索 Blink...

# Optional typed runtime profile. Product wrappers set this themselves.
# BLINK_LOCAL_CONFIG_PROFILE=browser-en-kokoro
# BLINK_LOCAL_CONFIG_PROFILES_PATH=

BLINK_LOCAL_LANGUAGE=zh
BLINK_LOCAL_STT_BACKEND=mlx-whisper
# Leave BLINK_LOCAL_TTS_BACKEND unset to use the runtime policy:
# prefer local-http-wav when available, otherwise fall back to Kokoro.
# BLINK_LOCAL_TTS_BACKEND=kokoro
BLINK_LOCAL_STT_MODEL=mlx-community/whisper-medium-mlx
BLINK_LOCAL_TTS_VOICE_ZH=zf_xiaobei
BLINK_LOCAL_TTS_VOICE_EN=af_heart
# BLINK_LOCAL_TTS_VOICE=
BLINK_LOCAL_XTTS_BASE_URL=http://127.0.0.1:8000
BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001
BLINK_LOCAL_ALLOW_BARGE_IN=
BLINK_LOCAL_MELO_HOST=127.0.0.1
BLINK_LOCAL_MELO_PORT=8001
BLINK_LOCAL_MELO_DEVICE=cpu
BLINK_LOCAL_MELO_SPEAKER_ZH=ZH
BLINK_LOCAL_MELO_SPEAKER_EN=EN-US
BLINK_LOCAL_MELO_SPEED=1.0
BLINK_LOCAL_AUDIO_INPUT_DEVICE=
BLINK_LOCAL_AUDIO_OUTPUT_DEVICE=
BLINK_LOCAL_HOST=127.0.0.1
BLINK_LOCAL_PORT=7860
BLINK_LOCAL_BROWSER_VISION=1
# Deprecated no-op for native voice. Use BLINK_LOCAL_CAMERA_SOURCE=macos-helper
# only for the English Kokoro helper path.
BLINK_LOCAL_VOICE_VISION=0
BLINK_LOCAL_CAMERA_SOURCE=none
# BLINK_LOCAL_CAMERA_HELPER_APP=native/macos/BlinkCameraHelper/build/BlinkCameraHelper.app
# BLINK_LOCAL_CAMERA_HELPER_STATE_DIR=
BLINK_LOCAL_VISION_MODEL=vikhyatk/moondream2
BLINK_LOCAL_CAMERA_DEVICE=0
BLINK_LOCAL_CAMERA_FRAMERATE=1
BLINK_LOCAL_CAMERA_MAX_WIDTH=640

# Optional hybrid demo LLM lane:
# OPENAI_API_KEY=
# BLINK_LOCAL_LLM_PROVIDER=openai-responses
# BLINK_LOCAL_OPENAI_RESPONSES_MODEL=gpt-5.4-mini
# BLINK_LOCAL_DEMO_MODE=1
# BLINK_LOCAL_LLM_MAX_OUTPUT_TOKENS=120
# BLINK_LOCAL_OPENAI_RESPONSES_SERVICE_TIER=
```

English fallback:

```dotenv
BLINK_LOCAL_LANGUAGE=en
BLINK_LOCAL_TTS_BACKEND=kokoro
BLINK_LOCAL_TTS_VOICE_EN=af_heart
```

Voice resolution order is:

1. CLI `--tts-voice`
2. `BLINK_LOCAL_TTS_VOICE_ZH` or `BLINK_LOCAL_TTS_VOICE_EN`
3. `BLINK_LOCAL_TTS_VOICE`
4. backend/language default

Use the language-specific env vars for normal operation. Keep `BLINK_LOCAL_TTS_VOICE`
only as a backward-compatible generic fallback.

`local-http-wav` is the exception: Blink does not reuse the Kokoro/Piper
voice env vars for that backend. Leave the speaker unset to use the sidecar
default, or pass an explicit speaker that appears in `GET /voices`. If the
sidecar exposes `/voices` and the configured speaker is stale, Blink now
warns and falls back to the sidecar default instead of failing the first reply.

For Chinese browser runtime, `local-http-wav` is also the first automatic choice
when it is healthy and no backend was pinned explicitly. Native voice is the
English Kokoro path and does not auto-switch to `local-http-wav`.

Text and speech now use different built-in defaults:

- `run-local-chat`: developer-friendly Chinese text output
- `run-local-voice` and `run-local-browser`: shorter speech-safe Chinese

If you set `OLLAMA_SYSTEM_PROMPT`, that shared override wins for lower-level
local CLI calls. The English native wrappers set
`BLINK_LOCAL_IGNORE_ENV_SYSTEM_PROMPT=1`, so
`./scripts/run-local-voice-en.sh` and
`./scripts/run-local-voice-macos-camera-en.sh` use the built-in English speech
prompt instead of inheriting a Chinese prompt from `.env`. A deliberate
`--system-prompt ...` passed to either wrapper still wins.

## What Is Actually Local

See [`LOCAL_CAPABILITY_MATRIX.md`](./LOCAL_CAPABILITY_MATRIX.md) for the detailed split between:

1. what runs locally now
2. what has been made locally runnable by this repo workflow
3. what still depends on external services or infrastructure

## First-Run Downloads

Some local assets are downloaded on demand unless you prefetch them:

- Ollama model `qwen3.5:4b`
- MLX Whisper model weights
- Kokoro model and voice assets when using Kokoro
- Piper voice files when using Piper
- Moondream weights when `--with-vision` is enabled

XTTS and `local-http-wav` are different: they use an external local HTTP service, so there is no in-repo TTS asset cache to prefetch. The recommended MeloTTS sidecar has its own isolated environment and model cache outside the main Blink package.

## TTS Evaluation

Use the local evaluation harness to generate deterministic zh/en WAV files for manual listening:

```bash
./scripts/eval-local-tts.sh
./scripts/eval-local-tts.sh --backend kokoro --all-kokoro-zh-voices
./scripts/eval-local-tts.sh --backend local-http-wav --language zh --voice-zh speaker-a
```

Outputs are written to `artifacts/tts-eval/`, which is intentionally ignored by git.

This harness is for comparing:

- Chinese versus English routing
- Kokoro Mandarin voice variants
- the bootstrap path versus a stronger sidecar path such as MeloTTS
- current in-repo backends versus a future higher-quality local HTTP speech server

`local-http-wav` is the official replacement seam for higher-quality TTS
engines. External reference checkouts such as CosyVoice are design material
only; they are not packaged runtime code. The MeloTTS sidecar server lives
outside `src/blink` for the same reason: it should not drag conflicting speech
dependencies into the main package.

## MeloTTS Sidecar

This repo now includes a repo-owned MeloTTS HTTP-WAV server outside `src/blink`. It keeps Blink on the stable
`local-http-wav` contract while isolating MeloTTS and its dependency stack in a separate Python `3.11` environment.

See [`docs/chinese-conversation-adaptation.md`](./docs/chinese-conversation-adaptation.md) for the broader reasons behind this design.

Bootstrap the isolated environment:

```bash
./scripts/bootstrap-melotts-reference.sh
```

That bootstrap clones a fresh upstream Melo checkout into `docs/MeloTTS-reference/vendor/MeloTTS`,
applies the repo-owned zh/en Apple Silicon patch layer, installs the tested runtime, and prefetches
the main sidecar assets so the first real voice session is not blocked on stale packaging or large model downloads.

Start the server:

```bash
./scripts/run-melotts-reference-server.sh
```

Or launch browser Blink already pointed at it:

```bash
./scripts/run-local-browser-melo.sh
```

This is the primary Chinese-quality browser path in the repo and is a peer to
the English `browser-en-kokoro` browser path.

`./scripts/run-local-browser-melo.sh` preserves the `browser-zh-melo` camera
default, so it should come up as browser voice + local vision unless you
explicitly disable `BLINK_LOCAL_BROWSER_VISION`.

`./scripts/run-local-browser-kokoro-en.sh` now preserves the same browser vision
default for `browser-en-kokoro`; use `--no-vision` or
`BLINK_LOCAL_BROWSER_VISION=0` only when you intentionally want audio-only
English browser mode.

For daily browser use, that wrapper keeps camera access enabled but disables
continuous background perception by default (`BLINK_LOCAL_CONTINUOUS_PERCEPTION=0`).
Explicit camera questions still use the bounded `fetch_user_image` path, while
the browser avoids spending CPU on constant visual analysis during normal voice
turns. Set `BLINK_LOCAL_CONTINUOUS_PERCEPTION=1` only when you are intentionally
testing background visual awareness.

Those launcher wrappers now own the sidecar process they spawn. If you stop the
wrapper with `Ctrl+C`, it should cleanly stop both the Blink runtime and the
Melo sidecar it started instead of leaving stale helper processes behind.

The Melo browser wrapper also writes a local runtime log for each run under
`artifacts/runtime_logs/` and updates
`artifacts/runtime_logs/latest-browser-melo.log`. Use that latest log together
with `/api/runtime/voice-metrics` when separating WebRTC microphone stalls,
STT waiting, camera stale-frame state, and TTS sidecar health. The runtime log
directory is local-only and ignored by git.

## CosyVoice Sidecar

This repo now includes an optional CosyVoice adapter that keeps Blink on the stable
`local-http-wav` contract while proxying to a separately running CosyVoice server.

If you want to start a local CosyVoice reference source directly instead of
managing that sidecar by hand:

```bash
./scripts/bootstrap-cosyvoice-reference.sh
./scripts/run-local-browser-cosyvoice.sh
```

That path uses an external CosyVoice reference checkout, starts the reference
FastAPI server locally, starts the Blink adapter, and then launches the
browser/WebRTC Blink app against `local-http-wav`. The reference source is
optional and is not committed in the source-first repository.

Start the adapter:

```bash
./scripts/run-local-cosyvoice-adapter.sh
```

Then point Blink at it:

```bash
BLINK_LOCAL_TTS_BACKEND=local-http-wav \
BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001 \
./scripts/run-blink-voice.sh
```

The full manual is in [CHINESE_TTS_UPGRADE.md](./CHINESE_TTS_UPGRADE.md).

This is still optional. The default in-repo Mandarin path remains `kokoro`, which
is the pragmatic bootstrap path for local MBP development. If spoken Chinese still
sounds accented after the prompt and normalization improvements, that remaining
limitation is the TTS backend itself, not the local Blink pipeline. The first upgrade to try is MeloTTS via `local-http-wav`.

The doctor command will tell you whether those caches are already present when the path is deterministic.

## Troubleshooting

`error: Could not reach Ollama...`

- Start the server with `ollama serve`
- Verify the endpoint with `curl http://127.0.0.1:11434/v1/models`

`Model 'qwen3.5:4b' is not available...`

- Pull it with `ollama pull qwen3.5:4b`

`error: Chinese XTTS voice requires a reachable XTTS-compatible server...`

- Start the XTTS server before running native voice or browser voice
- Default endpoint: `http://127.0.0.1:8000`
- Or switch to another local backend with `BLINK_LOCAL_TTS_BACKEND=local-http-wav`

`Could not reach the local HTTP WAV TTS server...`

- If you are following the recommended path, start MeloTTS first with `./scripts/run-melotts-reference-server.sh`
- Or use the one-command browser wrapper `./scripts/run-local-browser-melo.sh`
- If you are not using MeloTTS, verify your own `/tts` sidecar and its base URL

Blink hears me but answers very slowly in the browser

See the concise regression runbook in [`docs/debugging.md`](./docs/debugging.md)
if this appears as "first answer works, second answer fails or sounds wrong."

- Check `curl http://127.0.0.1:7860/api/runtime/voice-metrics` first. If
  `microphone_state` is `receiving`, `stt_state` is `transcribed`, and
  `stt_waiting_too_long` is false, the bottleneck is probably not MLX Whisper.
- Inspect `chunk_count` and `average_chunks_per_response`. High values with
  MeloTTS usually mean the response is being split into too many short
  `local-http-wav` synthesis jobs.
- The expected Melo path buffers normal assistant turns into moderate bounded
  `local-http-wav` chunks. If audio stops early, check for a regression in the
  local HTTP WAV TTS context timeout before changing STT, model, or camera
  settings. If the second reply sounds distorted or faded, inspect the bounded
  Melo request logs for `chars`, `bytes`, `audio_ms`, and `request_ms`.
- Do not "fix" this by changing browser media capture, re-enabling automatic
  WebRTC renegotiation, or switching STT/model first. The repeated regression
  was a mix of camera frame-rate pressure, too many tiny Melo chunks, then an
  overcorrection to overlarge chunks.
- Close duplicate `/client/` tabs or reload the page after asset changes. The
  operator workbench uses bounded polling, but old open tabs can keep running
  older JavaScript until they are reloaded.

I only want the local MBP workflow to run without extra voice infrastructure

- Keep `BLINK_LOCAL_TTS_BACKEND=kokoro`
- Use `./scripts/run-local-voice-en.sh` for the camera-free English native
  voice path, or the browser Melo wrapper for the Chinese camera path

The local Mandarin voice path still uses an old English voice

- Check `BLINK_LOCAL_TTS_BACKEND`; the default local Chinese path is `kokoro`
- Clear old `BLINK_LOCAL_TTS_VOICE`, `BLINK_LOCAL_TTS_VOICE_ZH`, or `BLINK_LOCAL_TTS_VOICE_EN`
  overrides if they point to the wrong voice
- Use `BLINK_LOCAL_LANGUAGE=en` only when you want the English fallback flow

The assistant started replying in English, but the spoken voice still sounds Chinese

- Speech language is fixed per session
- Restart the session with `BLINK_LOCAL_LANGUAGE=en`
- Do not expect the TTS service to hot-swap because of an in-conversation instruction

`ollama run qwen3.5:4b "hello"` prints a long reasoning trace before the answer

- That is Ollama's native CLI behavior for thinking-capable models
- The Blink local CLIs suppress that by default with `reasoning_effort=none`
- If you still get no assistant text from `./scripts/run-blink-chat.sh`, verify Ollama itself with `curl http://127.0.0.1:11434/v1/models`

The local voice stack speaks raw `*` or markdown markers

- The shared local TTS path now strips markdown before Kokoro or Piper speak it
- If you still hear formatting artifacts, verify you are on the updated local CLI entry points instead of an older example script

The browser has camera permission, but the assistant says it cannot see the camera

- `browser-zh-melo` enables vision by default
- `browser-en-kokoro` also enables vision by default
- `./scripts/run-local-browser-melo.sh` and
  `./scripts/run-local-browser-kokoro-en.sh` carry that default instead of
  silently dropping to audio-only browser mode
- Both primary browser wrappers leave continuous perception off by default to
  keep voice turns reactive while the camera is open. This should not block
  explicit camera questions; those should still call `fetch_user_image`.
- SmallWebRTC input capture honors the requested framerate for camera frames.
  If camera-open sessions become slow again, check for a regression that is
  converting every incoming WebRTC video frame instead of emitting the bounded
  capture rate.
- Fresh browser server startup records the camera as disconnected until a new
  `/client/` connection arrives, so doctor output should not treat the previous
  run's stalled camera as current live state.
- If Blink never logs `Track video received`, reload `/client/` after granting
  camera permission and check whether another app already owns the camera
- Install the optional vision stack with `--with-vision` if camera inspection is not available yet
- Use `./scripts/run-local-browser.sh --vision` only when you want to bypass the default Melo-backed launcher behavior
- Make sure you opened the browser UI at `http://127.0.0.1:7860/client/`, not a TTS sidecar port
- If you changed browser permissions or were already on an older open page,
  reload the `/client/` tab once before debugging the backend
- If the browser backend log does not show `Track video received`, this is still a WebRTC or permission problem
- If the log shows `Track video received` but the answer is vague or says the text is blurry, the issue is now in the local camera-analysis path rather than camera access itself
- In that case, debug `src/blink/cli/local_browser.py` and the local vision prompt/result path before changing TTS

`pyaudio` or `portaudio` build errors

- Native voice requires `brew install portaudio`
- The text and browser profiles do not require local audio

`blink.runner.run` fails because FastAPI or runner dependencies are missing

- Install the browser profile: `./scripts/bootstrap-blink-mac.sh --profile browser`

## Canonical Brain-Core Proof Lane

For headless brain-core work, bootstrap only the base development dependencies:

```bash
uv sync --python 3.12 --group dev
```

That proof lane does not require browser, audio, robot-head, provider, or
vision extras.

Use the canonical repo-owned entrypoint:

```bash
./scripts/test-brain-core.sh
```

Useful lane splits:

```bash
./scripts/test-brain-core.sh --lane fast
./scripts/test-brain-core.sh --lane proof
./scripts/test-brain-core.sh --lane fuzz-smoke
./scripts/test-brain-core.sh --lane atheris
```

Interpretation:

- `fast` is the always-run headless lane for import hygiene, replay, packet
  proofs, continuity evals, planning, and active-state unit coverage
- `proof` is the maintained property plus stateful lane
- `fuzz-smoke` keeps deterministic harness coverage in the default stack
- `atheris` is opt-in or scheduled only on libFuzzer-capable machines

If the proof lane fails because of browser, audio, or hardware setup, that is a
bug in the proof surface. Use `./scripts/doctor-blink-mac.sh --profile ...`
only for optional runtime workflows, not to excuse headless brain-core
collection failures.

## Canonical Embodied-Core Proof Lane

For the current embodied and perception slice, use the dedicated repo-owned
entrypoint:

```bash
./scripts/test-embodied-core.sh
```

Useful lane splits:

```bash
./scripts/test-embodied-core.sh --lane fast
./scripts/test-embodied-core.sh --lane proof
```

Interpretation:

- `fast` is the maintained regression lane for perception broker behavior,
  embodied runtime-shell slices, robot-head policy/runtime sequencing, and
  bounded embodied capability dispatch
- `proof` is the maintained property plus stateful lane for scene-world,
  multimodal autobiography, and policy-coupled embodied-adjacent flows

## Curated Memory And Personality Seed

For abundant local setup, use the preview-first curated JSON importer. This is
local-only and does not call OpenAI, ChatGPT, Ollama, or any external model.
It writes through governed memory and typed behavior controls instead of hidden
prompt or persona mutations.

The built-in Witty Sophisticated preset is the fastest safe path for making
Blink more humorous, vivid, sophisticated, and character-rich without fake
autobiography:

```bash
blink-memory-persona-ingest \
  --preset witty-sophisticated \
  --dry-run \
  --report preview.json
```

Apply still requires the exact approved preview:

```bash
blink-memory-persona-ingest \
  --preset witty-sophisticated \
  --apply \
  --approved-report preview.json
```

Example `seed.json`:

```json
{
  "schema_version": 1,
  "language": "zh",
  "user_profile": {
    "name": "小周",
    "role": "本地产品设计者",
    "origin": "上海"
  },
  "preferences": {
    "likes": ["机器人", "简洁解释"],
    "dislikes": ["长篇寒暄"]
  },
  "relationship_style": {
    "interaction.style": ["warm but concise"],
    "interaction.misfire": ["too much preamble"]
  },
  "teaching_profile": {
    "teaching.preference.mode": ["walkthrough"],
    "teaching.preference.analogy_domain": ["physics"],
    "teaching.history.helpful_pattern": ["stepwise decomposition"]
  },
  "behavior_controls": {
    "response_depth": "concise",
    "memory_use": "continuity_rich",
    "initiative_mode": "proactive",
    "evidence_visibility": "rich",
    "correction_mode": "rigorous",
    "explanation_structure": "walkthrough"
  }
}
```

Preview first, then apply the exact approved preview:

```bash
blink-memory-persona-ingest --seed seed.json --dry-run --report preview.json
blink-memory-persona-ingest --seed seed.json --apply --approved-report preview.json
```

For the browser default local profile, keep the default runtime scope or pass
`--client-id local_primary` if you want to target that user explicitly.

The browser operator workbench exposes the same Witty Sophisticated seed under
Controls. It shows a public-safe preview, requires confirmation before apply,
and reports written/no-op/rejected entries inline. Browser apply is rejected
unless an active runtime store is available and the posted approved preview
matches the generated seed hash.

## Development Loop

Common repo commands:

```bash
./scripts/test-brain-core.sh
uv run pytest
uv run ruff check
uv run ruff format --check
```

Opt-in browser smoke:

```bash
uv run python -m playwright install chromium
BLINK_RUN_BROWSER_E2E=1 uv run pytest -m browser_e2e
```

Recommended reading:

- [`tutorial.md`](./tutorial.md)
- [`AGENTS.md`](./AGENTS.md)
- [`examples/README.md`](./examples/README.md)

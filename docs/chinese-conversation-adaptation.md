# Chinese Conversation Adaptation Record

This document is the authoritative technical record of the work done to adapt
Blink's local product path for effective Chinese-language conversation.

It explains what changed, why it changed, which components were affected, what
problems were solved, what tradeoffs remain, and what future developers should
preserve when extending the system.

## Scope

The work documented here covers the repo's local-first Chinese conversation
path:

- terminal chat
- native mic and speaker voice
- browser and WebRTC voice
- Chinese speech quality evaluation
- the external local TTS seam used for stronger Mandarin backends

It does not redesign the core Blink frame architecture. The adaptation effort
focuses on making the existing local product path behave better for Chinese
conversation.

## Executive Summary

The repo was shifted from a generic local demo posture to a Chinese-first local
conversation posture.

The main architectural decisions are:

- Simplified Chinese is the default local interaction language.
- Kokoro remains the zero-setup bootstrap TTS path.
- MeloTTS behind `local-http-wav` is the primary recommended Chinese-quality
  upgrade path.
- Chinese voice and browser runtime now auto-prefer a healthy Melo sidecar when
  no backend is explicitly pinned.
- the primary browser product paths are now two peer lanes:
  `browser-zh-melo` for Chinese MeloTTS with camera, and `browser-en-kokoro`
  for English Kokoro with camera/Moondream and without MeloTTS sidecar load.
- CosyVoice remains available, but as an advanced optional secondary sidecar.
- Speech remains fixed per session. In-session TTS language switching is still
  not part of the local product path.
- Native local voice now defaults to protected playback so the assistant does
  not cut itself off because of speaker bleed.

The bilingual actor runtime now makes these browser paths observable through
shared public state and event contracts: `/api/runtime/actor-state`,
`/api/runtime/actor-events`, actor trace replay, Actor Surface v2, and the
deterministic bilingual actor bench. Chinese/Melo and English/Kokoro must keep
the same runtime-state schema, camera scene contract, interruption policy,
active-listening surface, memory/persona summary shape, and release gate.

## Original Problems

The adaptation effort started from a practical diagnosis rather than from a
framework rewrite.

The main problems were:

1. The repo story was misaligned with the actual goal.
   The old local workflow emphasized "what runs first" rather than "what makes
   Chinese voice interaction usable."

2. Chinese speech content was not shaped for speech.
   Text chat and speech workflows were too similar, which meant Chinese spoken
   output often contained raw symbols, paths, URLs, version strings, and mixed
   technical fragments that sounded unnatural.

3. The default Mandarin TTS path was runnable but not the quality target.
   Kokoro is useful for local bootstrap, but not the final Mandarin quality bar.

4. Stronger Mandarin backends were operationally fragile.
   Upstream MeloTTS packaging, model bootstrap, and Apple Silicon runtime
   behavior created enough friction that it was not a reliable local upgrade
   path without repo-owned hardening.

5. Native local audio could self-interrupt.
   In practice, some Chinese voice responses were cut off after a second or two.
   The root issue was not a hard TTS cap; it was native local interruption
   behavior combined with speaker bleed and the lack of echo cancellation in the
   local audio transport.

## What Changed

### 1. Repo Goal And Narrative Reset

The repo documentation and local workflow guidance were revised so they tell one
consistent story:

- Chinese voice interaction quality is the main local product target.
- Kokoro is the bootstrap path, not the end-state Mandarin quality target.
- MeloTTS via `local-http-wav` is the first recommended Chinese-quality path.
- Chinese browser runtime prefers Melo automatically when it is available, while
  keeping Kokoro as the regression-safe fallback.
- `./scripts/run-local-browser-melo.sh` and
  `./scripts/run-local-browser-kokoro-en.sh` are equal primary browser/WebRTC
  lanes for Chinese MeloTTS and English Kokoro respectively.
- The default native local voice path is English-only Kokoro and intentionally
  does not use camera, Moondream, or MeloTTS.
- The separate `./scripts/run-local-voice-macos-camera-en.sh` path is also
  English-only Kokoro; it uses `BlinkCameraHelper.app` for app-owned macOS
  camera permission and runs Moondream only on explicit image-tool requests.
- the `browser-zh-melo` and `browser-en-kokoro` launchers enable camera support
  by default while the lower-level browser CLI keeps explicit `--vision` and
  `--no-vision` controls
- The core package should stay clean of conflicting TTS dependencies.

Affected surfaces include:

- [`README.md`](../README.md)
- [`AGENTS.md`](../AGENTS.md)
- [`LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md)
- [`CHINESE_TTS_UPGRADE.md`](../CHINESE_TTS_UPGRADE.md)
- [`env.local.example`](../env.local.example)

### 2. Split Text Chat Defaults From Spoken Defaults

The local product path now distinguishes between text-first and speech-first
assistant behavior.

Affected component:

- [`src/blink/cli/local_common.py`](../src/blink/cli/local_common.py)

What changed:

- text chat keeps a developer-friendly Chinese default
- voice and browser flows use a speech-safe prompt
- the speech prompt avoids markdown, code formatting, raw URLs, shell syntax,
  and dense formatting
- explanatory spoken answers are allowed to be fuller than before instead of
  being compressed into unnaturally short replies

Why it changed:

- good text output and good spoken output have different constraints
- Chinese TTS becomes less usable when the LLM emits raw technical formatting
- overly aggressive brevity made spoken explanations incomplete

### 3. Deterministic Chinese Speech Normalization

The repo now includes a stronger rule-based Chinese speech normalization layer.

Affected component:

- [`src/blink/cli/local_common.py`](../src/blink/cli/local_common.py)

What changed:

- markdown markers are removed
- URLs are replaced with a neutral spoken placeholder
- file paths are normalized to a spoken placeholder
- versions such as `v2.3.1` are made speakable
- dates and times are rewritten into natural Chinese forms
- units such as `120ms` and `3s` are normalized
- technical shorthand such as `API`, `HTTP`, `CPU`, `TTS`, and `STT` is
  paraphrased into safer spoken Chinese
- mixed Chinese-English fragments and alphanumeric identifiers are split more
  naturally
- repeated ASCII punctuation is collapsed

Why it changed:

- technical Chinese conversation often contains exactly the kinds of tokens that
  baseline Mandarin TTS handles poorly
- the fastest reliable improvement was deterministic pre-TTS cleanup rather than
  a heavy new NLP dependency

### 4. MeloTTS Sidecar On The Existing `local-http-wav` Seam

The repo now includes a repo-owned MeloTTS reference server and bootstrap flow.

Affected components:

- [`local_tts_servers/melotts_http_server.py`](../local_tts_servers/melotts_http_server.py)
- [`local_tts_servers/melotts_reference.py`](../local_tts_servers/melotts_reference.py)
- [`scripts/bootstrap-melotts-reference.sh`](../scripts/bootstrap-melotts-reference.sh)
- [`scripts/run-melotts-reference-server.sh`](../scripts/run-melotts-reference-server.sh)
- [`scripts/run-local-browser-melo.sh`](../scripts/run-local-browser-melo.sh)
- [`docs/MeloTTS-reference/README.md`](./MeloTTS-reference/README.md)
- [`docs/MeloTTS-reference/runtime-requirements.txt`](./MeloTTS-reference/runtime-requirements.txt)

What changed:

- the repo owns a Melo HTTP-WAV sidecar outside `src/blink`
- the sidecar exposes `/healthz`, `/voices`, and `/tts`
- Blink continues to use the existing `local-http-wav` backend
- the sidecar returns `24 kHz` mono WAV
- Melo models are cached by language and device
- the bootstrap creates an isolated Python `3.11` environment
- the repo patches upstream Melo for the zh/en Apple Silicon local path

Why it changed:

- MeloTTS is a better fit for Chinese and mixed Chinese-English speech quality
  than the zero-setup bootstrap path
- upstream Melo dependencies conflict with the main repo's transformer stack
- sidecar isolation keeps the main package dependency graph clean

### 5. Native Voice Playback Reliability Fix

The most important runtime fix after the TTS quality work was preventing native
local voice from cutting itself off.

Affected components:

- [`src/blink/cli/local_common.py`](../src/blink/cli/local_common.py)
- [`src/blink/cli/local_voice.py`](../src/blink/cli/local_voice.py)

What changed:

- native local voice now defaults to muting user-turn detection while the
  assistant is speaking
- `--allow-barge-in` and `BLINK_LOCAL_ALLOW_BARGE_IN=1` were added as an
  explicit opt-in
- runtime status output now reports whether barge-in is on or off

Why it changed:

- the local native path uses [`LocalAudioTransport`](../src/blink/transports/local/audio.py),
  which has no built-in echo cancellation
- when speaker output leaked back into the mic, VAD could interpret the bot's
  own voice as a new user turn
- once an interruption is triggered, the output path must clear queued playback
  and abort the active native PyAudio write; clearing only the async queue can
  leave an already-submitted device write draining

Important technical conclusion:

- the "two-second reply" symptom was not a hard limit in Melo or in the
  `local-http-wav` seam
- deterministic TTS evaluation already proved the pipeline could synthesize
  multi-second Chinese WAVs
- the live failure mode was interruption behavior in native local audio

### 6. Doctor And Evaluation Guidance Now Reflect Chinese Reality

Affected components:

- [`src/blink/cli/local_doctor.py`](../src/blink/cli/local_doctor.py)
- [`src/blink/cli/local_tts_eval.py`](../src/blink/cli/local_tts_eval.py)

What changed:

- doctor warns when Chinese sessions are using Kokoro so developers understand
  that the path is valid for bootstrap and debugging but not the target quality
  path
- the evaluation harness now uses Chinese technical phrases, dates, times,
  versions, symbols, and a mixed-language diagnostic phrase
- evaluation output explicitly encourages comparison between Kokoro and
  `local-http-wav` sidecars such as MeloTTS

Why it changed:

- the old guidance made it too easy to confuse "runnable" with "good enough"
- Chinese voice quality needed a repeatable listening workflow, not just ad hoc
  impressions

### 7. Browser Camera Analysis Was Moved To An On-Demand Local Vision Path

Affected component:

- [`src/blink/cli/local_browser.py`](../src/blink/cli/local_browser.py)

What changed:

- the lower-level browser CLI still exposes `--vision`, but the canonical
  `./scripts/run-blink-browser.sh` launcher now enables the camera path by
  default
- the Melo browser wrapper `./scripts/run-local-browser-melo.sh` now carries
  that same default through to `run-local-browser.sh` instead of silently
  starting an audio-only browser session
- the browser runtime now caches the latest camera frame locally instead of
  speaking raw vision output directly
- when the user asks a camera question, the assistant calls
  `fetch_user_image`
- `fetch_user_image` analyzes the cached frame through local Moondream and
  returns the result as tool output
- the vision prompt is now derived from the actual last user utterance rather
  than trusting an LLM-rewritten tool argument
- browser vision retries once when the first local vision result is obviously
  garbled or too weak

Why it changed:

- camera permission and camera-answer quality are different problems
- the old browser vision path could receive valid video frames and still answer
  poorly because raw vision text was being used too directly
- local camera questions such as “我手里拿着什么” or “我身后有什么” are more
  reliable when the semantic intent comes from the real user utterance

Important technical conclusion:

- if the browser log does not show a video track, the problem is still session
  bootstrap or permission
- if the browser log does show a video track, but the answer is weak, the next
  layer to debug is local vision quality rather than TTS

## Problems Solved

The adaptation effort solved or materially improved these problems:

- the repo now explains the Chinese-first local product path clearly
- spoken Chinese is shaped differently from text chat
- Mandarin TTS receives safer, more speakable text
- stronger Chinese-quality speech is available through a maintained sidecar path
- Melo bootstrap is operationally easier on Apple Silicon than raw upstream
  setup
- native local voice no longer defaults to self-interrupting playback
- browser camera inspection now follows a more controlled local vision path
  instead of speaking raw vision output directly

## Components Affected

Core local CLI behavior:

- [`src/blink/cli/local_common.py`](../src/blink/cli/local_common.py)
- [`src/blink/cli/local_voice.py`](../src/blink/cli/local_voice.py)
- [`src/blink/cli/local_browser.py`](../src/blink/cli/local_browser.py)
- [`src/blink/cli/local_doctor.py`](../src/blink/cli/local_doctor.py)
- [`src/blink/cli/local_tts_eval.py`](../src/blink/cli/local_tts_eval.py)

External TTS seam and sidecars:

- [`src/blink/services/local_http_wav/tts.py`](../src/blink/services/local_http_wav/tts.py)
- [`src/blink/cli/local_cosyvoice_adapter.py`](../src/blink/cli/local_cosyvoice_adapter.py)
- [`local_tts_servers/melotts_http_server.py`](../local_tts_servers/melotts_http_server.py)
- [`local_tts_servers/melotts_reference.py`](../local_tts_servers/melotts_reference.py)

Operational docs and guidance:

- [`README.md`](../README.md)
- [`LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md)
- [`CHINESE_TTS_UPGRADE.md`](../CHINESE_TTS_UPGRADE.md)
- [`tutorial.md`](../tutorial.md)
- [`AGENTS.md`](../AGENTS.md)

Verification:

- [`tests/test_local_chat.py`](../tests/test_local_chat.py)
- [`tests/test_local_doctor.py`](../tests/test_local_doctor.py)
- [`tests/test_local_workflows.py`](../tests/test_local_workflows.py)
- [`tests/test_local_tts_eval.py`](../tests/test_local_tts_eval.py)
- [`tests/test_melotts_http_server.py`](../tests/test_melotts_http_server.py)
- [`tests/test_melotts_reference.py`](../tests/test_melotts_reference.py)

## Remaining Tradeoffs

Some tradeoffs are intentional and should not be treated as accidental bugs.

1. Kokoro remains the bootstrap default and fallback.
   This is a usability and regression-safety choice for local setup, not a
   claim that Kokoro is the best Mandarin path.

2. Native barge-in is off by default.
   This improves reliability on open-speaker laptop setups, but it means the
   native path does not prioritize "talk over the bot at any time" as the
   default interaction mode.

3. Speech stays fixed per session.
   This keeps local routing predictable, but it means a Chinese session cannot
   hot-swap into English TTS mid-session.

4. Melo remains sidecar-only.
   This preserves dependency hygiene, but it also means an extra local process
   and an extra isolated environment.

5. Browser voice still depends on a healthy WebRTC session bootstrap.
   If the browser client never completes its WebRTC offer flow, the LLM and TTS
   pipeline will not start, regardless of whether Ollama and Melo are healthy.

6. Local browser camera quality still depends on the vision model and frame detail.
   Even with the on-demand camera-analysis path, small blurry text and weak
   lighting can still produce vague answers. That is a local vision limitation,
   not a Chinese TTS limitation.

### 2026-04-24 Live Browser Debug History

Operator-reported symptoms:

- Blink could not finish one complete spoken sentence.
- The browser camera suddenly went down.
- The program appeared to go down.

What the live checks showed:

- The backend process did not fully crash. `/api/runtime/stack`,
  `/api/runtime/voice-metrics`, and the operator snapshot endpoint still
  responded after the failure.
- MeloTTS was reachable through the `local-http-wav` sidecar, and `/healthz`
  stayed healthy on port `8001`.
- The browser media path did degrade. Logs showed microphone and camera
  SmallWebRTC stall warnings, and the doctor/audit surface reported
  `camera_frame_stale` with the visual track in a stalled/degraded state.
- Voice metrics showed many short chunks for a small number of responses, which
  matched the audible symptom that Blink sounded as if it could not complete a
  sentence.

Root cause and lesson:

- The voice policy was treating Chinese and English commas as hard spoken-flush
  boundaries. That made the browser path send clause-sized chunks to MeloTTS.
  The intended local Chinese path is sentence-based aggregation, not comma-based
  flushing.
- Browser camera failure must be diagnosed separately from backend availability.
  A WebRTC media track can stall while the LLM, TTS sidecar, FastAPI process,
  and operator APIs remain healthy.
- The `Perception broker loop error: bad parameter or other API misuse` message
  was not public-facing, but it was too vague for live debugging. The log now
  includes the exception type so future failures are easier to classify.

Changes landed from this debug pass:

- Removed comma and Chinese comma from the voice chunk boundaries in
  `src/blink/brain/processors.py`.
- Added regression tests proving Chinese and English commas do not flush short
  clauses before a sentence boundary.
- Kept public runtime payloads unchanged.
- Added clearer perception-broker exception logging without widening browser
  API payloads.

Follow-up hardening landed after the first debug pass:

- Browser disconnect now clears/persists camera state as disconnected instead
  of leaving the previous stale frame as the apparent live state.
- Disconnected and first-frame-waiting camera health no longer carries stale
  recovery-in-progress metadata from an earlier stall loop.
- Browser connect no longer implies a fresh camera frame before one has actually
  arrived; the visual health state reports waiting/degraded until the first
  frame is cached.
- Input/STT health now observes microphone frames and VAD before Whisper, then
  observes transcript/error completion after Whisper. The workbench can show
  "mic is receiving, STT is waiting" without exposing raw transcript text.
- Browser and operator voice-health, behavior-control, expression,
  teaching/knowledge, memory, practice, adapter, rollout, sim-to-real, and
  evidence payloads now reject malformed public state names, timestamps,
  hint lists, count-map keys, and reason-code shapes instead of echoing
  arbitrary diagnostic strings, stack trace markers, or local DB paths.
- The standalone browser model catalog, stack, expression, memory,
  behavior-control, memory/personality preset, rollout-action, and episode
  evidence endpoints use the same public-safe sanitizers for top-level fields
  and malformed controller/report results, so the browser API and aggregate
  operator snapshot now share the same boundary expectations.
- Follow-up browser API hardening made session-scoped WebRTC offers reject
  non-object JSON instead of crashing, removed raw live user-turn previews from
  rollout evidence, sanitized rejected model/memory/action identifiers, and made
  public booleans/floats strict so strings like `"yes"` or `"nan"` cannot
  masquerade as real health or rollout state.
- Browser doctor now labels old persisted visual-state projections as
  `historical_not_live`, so a previous crashed/stalled camera run is not
  confused with the currently running browser. Camera disconnect/reconnect also
  clears stale recovery-action and audio-track detail metadata from the presence
  projection.
- `/api/runtime/operator` now applies one final public-safe aggregate sanitizer
  after building the workbench snapshot. This preserves the compact section
  contract while dropping malformed prompt text, raw transcript fields, source
  refs, stack traces, local DB paths, and non-finite metrics even if a future
  section builder regresses.
- `./scripts/run-local-browser-melo.sh` writes timestamped local logs under
  `artifacts/runtime_logs/` with `latest-browser-melo.log` pointing to the most
  recent run; that log directory is ignored by git.

Additional media hardening landed after the automation-browser
permission/device failure, and was then narrowed again after the same live
camera/audio symptoms repeated:

- A cross-check against the `dream` and `main` branches showed the packaged
  SmallWebRTC browser UI still defaulted `enableCam` to false even when the
  backend launched with browser vision enabled. The `browser-zh-melo` bundle now
  defaults to `enableCam=true` and `enableMic=true`, passes those options
  explicitly at the root client mount, and bumps the main app bundle cache key
  so the UI requests the real camera track for the Melo + vision path.
- The experimental startup asset that wrapped `getUserMedia`, retried
  microphone-only capture, and posted client-media state was rolled back. The
  shipped `blink-media-autoplay.js` is passive again: it only retries playback
  for autoplay media elements and does not mutate media tracks, permissions, or
  device constraints.
- `/api/runtime/stack` still exposes the sanitized `browser_media` projection,
  and `/api/runtime/client-media` remains a bounded public endpoint, but the
  default startup asset no longer drives that endpoint.
- The local browser runtime now suppresses only the known benign
  `aioice` STUN `Transaction.__retry` invalid-state race that can fire after
  WebRTC close/renegotiation. Other loop exceptions still go through the normal
  handler.
- SmallWebRTC still reports confirmed backend microphone/camera receive-track
  stalls, but the transport no longer requests immediate peer renegotiation from
  the stall callback. Operators should refresh or reconnect `/client/` first
  when the browser preview remains visible but Blink no longer receives frames.
- After a live second-turn regression where the first answer was audible but the
  second user turn never reached STT, the local browser path also stopped
  server-triggered startup renegotiation and packaged-client automatic
  ICE-reconnect attempts. The stable recovery contract is now explicit page
  reload/reconnect until the browser renegotiation path has separate proof.
- Browser/WebRTC voice now uses protected playback by default through
  server-side turn muting while the bot is speaking. This suppresses false VAD
  interruptions from speaker bleed without patching `getUserMedia`, mutating
  browser media tracks, or triggering WebRTC renegotiation. Browser
  `--allow-barge-in` disables this protection only for echo-safe barge-in tests.
  MeloTTS health should still be diagnosed separately from interruption/flush
  counts.
- Camera health and perception fusion now tolerate malformed or missing frame
  timestamps and confidence values, so bad metadata cannot keep the visual
  health loop in a repeated `TypeError` warning cycle.
- The browser camera slowdown path was traced to SmallWebRTC ignoring the
  requested capture `framerate=1` and converting every incoming camera frame to
  RGB before the brain pipeline saw it. SmallWebRTC now rate-limits emitted
  input video frames before RGB conversion, and the Melo browser wrapper keeps
  continuous perception disabled by default while leaving camera access on.
  Explicit camera questions still go through the governed `fetch_user_image`
  path; background visual awareness can be re-enabled with
  `BLINK_LOCAL_CONTINUOUS_PERCEPTION=1` for focused tests.
- Browser startup now resets the persisted browser visual projection to
  `camera_disconnected` before a fresh `/client/` connection arrives. This keeps
  doctor output from reporting a previous run's stalled camera as the current
  live state while the new server is merely waiting for the browser to connect.
- The browser camera health manager is diagnostic-only by default. Stale or
  stalled camera frames produce bounded public state such as
  `camera_manual_reload_required`; they no longer trigger automatic capture or
  renegotiation actions against the active WebRTC session.
- A later slow-answer pass separated STT latency from TTS latency. Live metrics
  showed microphone input and STT completion were healthy, while the
  `local-http-wav` path produced 87 TTS-bound chunks for 9 responses and the
  operator workbench was polling five status/evidence endpoints every 2.5
  seconds from multiple open browser tabs. The workbench now uses a slower,
  non-overlapping poll loop with hidden-tab backoff and long-lived static
  catalog refreshes; the local browser API caches bursty public
  status/evidence reads briefly; and `local-http-wav` voice chunking now
  buffers normal responses until the end of the assistant turn, then sends
  moderate bounded sentence groups to MeloTTS. This avoids repeated
  per-sentence WAV fade-outs without handing Melo an overlong paragraph that can
  sound unstable on the next turn. The local HTTP WAV TTS adapter also uses a
  longer audio-context timeout so a slow Melo synthesis call does not cause an
  early `TTSStoppedFrame` before the returned WAV arrives. The Melo sidecar
  normalizes returned WAV peak level and logs only bounded request metrics
  (`chars`, `bytes`, `audio_ms`, `request_ms`) so second-turn speech faults can
  be debugged without storing raw text or audio.

Verification performed:

- `uv run pytest tests/test_brain_voice_metrics.py`
- `uv run pytest tests/test_brain_perception_broker.py`
- `uv run pytest tests/test_local_workflows.py`
- `uv run ruff check` on the touched Python files
- `git diff --check`

Recommended recovery workflow after this failure mode:

1. Confirm backend and sidecar health:
   `curl http://127.0.0.1:7860/api/runtime/stack`,
   `curl http://127.0.0.1:7860/api/runtime/voice-metrics`, and
   `curl http://127.0.0.1:8001/healthz`.
2. If the stack is healthy but camera state is stale or stalled, reload the
   browser client at `http://127.0.0.1:7860/client/` and grant microphone and
   camera permission again.
3. For stability-focused diagnosis, keep browser vision enabled but temporarily
   disable continuous background perception with
   `BLINK_LOCAL_CONTINUOUS_PERCEPTION=0`. Explicit `fetch_user_image` camera
   questions still work through the cached-frame path.
4. Treat repeated clause-sized Chinese speech as a voice chunking regression,
   not as a MeloTTS hard output limit. The expected `local-http-wav` path is now
   turn-level coalescing with moderate bounded chunks, not one HTTP WAV request
   per short sentence and not one overlong paragraph per reply.
5. If STT is healthy but answer audio is slow, inspect `chunk_count`,
   `average_chunks_per_response`, and operator polling in
   `artifacts/runtime_logs/latest-browser-melo.log`. High chunk counts with a
   healthy sidecar point at response/chunk coalescing rather than a broken mic.

### 2026-04-25 Browser Melo Second-Reply Regression

This regression repeated several symptoms that had previously been solved:

- the first spoken answer could work, but the second answer sounded abnormal or
  stopped early
- camera and microphone sometimes degraded together after a successful first
  turn
- the system felt slow once the camera was enabled
- debugging drifted between STT, camera, LLM, and MeloTTS because the log did
  not separate those layers clearly enough

Root causes and near-misses:

- SmallWebRTC was previously converting incoming camera frames before honoring
  the requested low capture framerate. With camera open, that made normal voice
  turns compete with unnecessary RGB conversion work. The stable rule is:
  rate-limit browser video frames before RGB conversion, and keep continuous
  perception disabled in the Melo browser wrapper unless explicitly testing
  background vision.
- The older voice policy sent too many small `local-http-wav` chunks to MeloTTS.
  Each short WAV had its own attack/release shape, so answers sounded like they
  faded or restarted at every clause.
- The first chunking fix went too far in the other direction: buffering an
  entire turn into very large Melo requests can make the second reply sound
  unstable. The stable rule is not "one paragraph per reply"; it is
  turn-aware coalescing into moderate bounded sentence groups.
- The generic `TTSService` audio-context timeout was too short for synchronous
  local HTTP WAV generation. Slow Melo synthesis could cause an early
  `TTSStoppedFrame` before the WAV arrived. The `local-http-wav` adapter now
  uses a longer timeout by default.
- The Melo sidecar had no bounded request/audio accounting, so there was no
  direct way to tell whether a bad second answer came from an oversized TTS
  request, slow synthesis, returned WAV size/duration, or browser playback. The
  sidecar and adapter now log only safe metrics: character count, WAV bytes,
  audio duration, request duration, sample rate, and channel count. They must
  not log raw transcript text, raw assistant text, prompts, SDP, audio bytes, or
  browser exception payloads.

Guardrails for future fixes:

1. Do not diagnose "no reply" or "bad second reply" as STT first. Check
   `/api/runtime/voice-metrics`, `/api/runtime/stack`, and
   `artifacts/runtime_logs/latest-browser-melo.log` before changing code.
2. Do not mutate browser media tracks, patch `getUserMedia`, or re-enable
   automatic WebRTC renegotiation to hide a speech problem.
3. Keep Melo behind `local-http-wav`; do not introduce an in-process Melo backend
   or pull Melo dependencies into the core package.
4. Keep the expected Melo chunk shape moderate and bounded. Too many chunks
   cause repeated fades; one overlong chunk can destabilize synthesis.
5. If camera-open sessions become slow again, inspect SmallWebRTC frame-rate
   gating before changing model, STT, or TTS settings.
6. If the browser preview remains visible but backend audio/video frames stop,
   reload/reconnect `/client/` as the supported recovery path. Automatic
   renegotiation needs separate proof before returning to the daily path.

## Browser-First Checkpoint: 2026-04-26

The repeated "AI can barely say a word" native English failure was not MLX
Whisper, Ollama, Kokoro, or camera pressure. The native wrapper had been forced
into barge-in mode; on laptop speakers, PyAudio heard Blink's own Kokoro output
through the Mac microphone and triggered self-interruption. Browser/WebRTC
remains the better daily-use media path because it has browser media handling,
permission UX, and echo-cancellation behavior that native PyAudio does not.

The follow-up rule is:

- focus product-quality voice work on browser/WebRTC first
- keep `./scripts/run-local-voice-en.sh` as a backend isolation lane
- treat `./scripts/run-local-browser-kokoro-en.sh` as the primary English
  Kokoro browser/WebRTC path without MeloTTS or continuous perception load
- treat `./scripts/run-local-browser-melo.sh` as the primary Chinese
  camera-capable MeloTTS browser/WebRTC path

## Constraints Future Developers Must Preserve

These constraints are now part of the local Chinese conversation architecture:

- do not add a direct in-process `melo` backend to `BLINK_LOCAL_TTS_BACKEND`
- keep stronger Mandarin engines behind `local-http-wav`
- keep Melo outside `src/blink`
- do not make external TTS mandatory for the default local text or voice path
- keep Chinese speech shaping deterministic and dependency-light unless there is
  strong evidence for a heavier approach
- do not regress the native protected-playback default unless proper echo
  cancellation is introduced
- keep browser vision on the cached-frame plus on-demand-analysis path unless a
  stronger local design clearly replaces it
- keep repo docs and skill guidance aligned with the implemented local workflow

## Follow-up Work

The highest-value follow-up items are:

1. Improve browser-session diagnostics.
   When the browser UI fails before sending a valid WebRTC offer, the AI
   pipeline never starts. That failure mode should be easier to diagnose from
   logs and UI messaging.

2. Add more explicit local runtime logging around connection setup.
   Chinese-conversation debugging currently benefits from direct terminal logs,
   but the browser path could surface better operator-facing diagnostics.

3. Evaluate whether a future echo-safe native path should support live barge-in
   by default again.
   That should only happen if the underlying transport path can avoid
   self-interruption reliably.

4. Continue improving the Chinese evaluation set and manual QA process.
   The TTS evaluation harness should remain the reproducible listening baseline
   for changes to the spoken Chinese path.

5. Evaluate whether browser camera mode needs a stronger local OCR or
   higher-resolution capture path.
   The current browser vision stack is much more debuggable now, but camera
   answers are still limited by local vision quality when the scene contains
   small or blurry text.

## Recommended Extension Workflow

When extending this system, future developers should work in this order:

1. read this record
2. read [`LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md)
3. read [`CHINESE_TTS_UPGRADE.md`](../CHINESE_TTS_UPGRADE.md)
4. inspect the current local CLI implementation in `src/blink/cli/`
5. run the targeted local tests before broad repo checks

If a change affects Chinese conversation quality, it should be evaluated against
all of these questions:

- does it help Chinese intelligibility
- does it preserve the `local-http-wav` seam
- does it keep bootstrap light enough for a local MacBook workflow
- does it avoid regressing native playback reliability
- does it keep English fallback behavior intact

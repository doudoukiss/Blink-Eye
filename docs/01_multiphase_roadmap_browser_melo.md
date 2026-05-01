# 01 — Multiphase Roadmap: Primary Browser/WebRTC Paths

## Primary routes

```bash
./scripts/run-local-browser-melo.sh
./scripts/run-local-browser-kokoro-en.sh
```

Open:

```text
http://127.0.0.1:7860/client/
```

## Roadmap overview

`browser-zh-melo` and `browser-en-kokoro` are equal primary browser/WebRTC
paths. Native English Kokoro remains a backend-isolation lane, not the primary
English product path.

### Phase 00 — Canonical Browser/WebRTC Launchers

**Goal:** Make the browser/WebRTC launchers unambiguous: `browser-zh-melo` for
Chinese MeloTTS with camera, and `browser-en-kokoro` for English Kokoro
with camera available by default.

**Build:**
- Print an explicit readiness line containing runtime=browser, transport=WebRTC, tts=local-http-wav/MeloTTS, vision state, continuous_perception state, and barge_in state.
- Preserve MeloTTS sidecar supervision and health checks in run-local-browser-melo.sh.
- Treat native English Kokoro as backend isolation only. Direct daily English UX
  work to `browser-en-kokoro`.
- Keep protected playback as the safe default. Do not force --allow-barge-in globally.
- Document both daily browser run commands and the shared client URL:
  `./scripts/run-local-browser-melo.sh`,
  `./scripts/run-local-browser-kokoro-en.sh`, then
  `http://127.0.0.1:7860/client/`.

**Acceptance:**
- Running the launcher prints a concise, unambiguous runtime summary.
- MeloTTS HTTP-WAV sidecar is either reused if healthy or started and supervised if absent.
- Browser vision defaults remain enabled for the Melo browser path unless explicitly disabled.
- Barge-in remains off by default unless the user passes --allow-barge-in or BLINK_LOCAL_ALLOW_BARGE_IN=1.
- bash -n passes for launcher scripts and existing local workflow tests pass.

### Phase 01 — Browser Performance Event Bus and BrowserInteractionState

**Goal:** Add a typed performance event stream and browser interaction state for WebRTC, MeloTTS, camera, vision, memory, and interruption status.

**Build:**
- Create a BrowserInteractionState dataclass or equivalent state holder with listening/thinking/speaking/looking/interrupted/waiting modes.
- Emit public-safe events from WebRTC connection lifecycle, client media reporting, VAD/STT processors, LLM start/end, Melo TTS start/end, camera frame buffering, fetch_user_image, and interruption frames.
- Expose state via /api/runtime/performance-state and events via /api/runtime/performance-events using SSE if practical, otherwise low-latency polling.
- Avoid leaking private chain-of-thought, raw prompt contents, credentials, or full memory store contents.
- Keep existing /api/runtime/expression, /api/runtime/voice-metrics, /api/runtime/client-media, and /api/runtime/memory endpoints working.

**Acceptance:**
- The browser runtime state changes correctly across connected, listening, heard, thinking, looking, speaking, interrupted, waiting, and error modes.
- A state snapshot can be retrieved without an active connection and reports unavailable/unreported media cleanly.
- Events are ordered, timestamped, scoped to a session/client, and safe to display to users.
- Existing runtime endpoints continue to pass tests.

### Phase 02 — Browser Feedback UI and Live Status Surface

**Goal:** Make the browser client visibly explain what Blink is doing at every moment.

**Build:**
- Add a compact user-facing status surface: connection, listening/heard/thinking/speaking/looking/waiting/interrupted.
- Show last final transcript, current assistant subtitle, camera state, MeloTTS state, and barge-in/protected-playback state.
- Show memory/persona used-this-turn summaries when available, not the entire memory store.
- Do not add another browser media acquisition path. Reuse the existing WebRTC client media connection.
- Add clear degraded-state messages: mic denied, camera denied, camera stale, Melo unavailable, LLM unavailable.

**Acceptance:**
- A user can tell whether Blink is listening, thinking, speaking, looking, interrupted, waiting, or degraded without reading terminal logs.
- Assistant text appears before or during Melo playback as subtitles.
- Camera and mic permission states are displayed accurately.
- The UI remains usable on a single laptop screen without becoming an operator-only dashboard.

### Phase 03 — WebRTC Turn-Taking, Interruption, and Echo Health

**Goal:** Make interruption reliable without repeating the native PyAudio self-interruption failure.

**Build:**
- Keep protected playback default. Treat --allow-barge-in as explicit advanced mode, not the daily default.
- Add an adaptive interruption policy that distinguishes real user speech from backchannels, coughs, and assistant audio leakage.
- Use browser/WebRTC media constraints and client telemetry to request echoCancellation, noiseSuppression, and autoGainControl where supported.
- Surface echo-safe status in the UI: protected, headphones recommended, barge-in armed, barge-in suppressed, interruption accepted.
- Drop stale Melo audio/text chunks after an accepted interruption and avoid speaking tool results from an interrupted turn.

**Acceptance:**
- Laptop-speaker default does not self-interrupt.
- When explicit barge-in is enabled in an echo-safe setup, user speech can stop Blink quickly and visibly.
- Short acknowledgments such as “嗯” or “对” do not always kill the assistant turn.
- Interruption attempts produce trace events: candidate, accepted/rejected, output flushed, listening resumed.

### Phase 04 — Melo Speech Director and Subtitles

**Goal:** Make Chinese MeloTTS speech performable, chunked, subtitle-aligned, and cancellation-safe.

**Build:**
- Add a speech-chunk abstraction with role, display_text, speak_text, interruptible, pause_after_ms, generation_id, and turn_id.
- Optimize chunking for Chinese punctuation and spoken comprehension, avoiding long uninterruptible monologues.
- Display subtitles before or at the same time as audio playback.
- Keep Melo sidecar HTTP-WAV contract stable and preserve existing health checks.
- Record time-to-first-subtitle, time-to-first-audio, chunk latency, queue depth, and stale-drop counts.

**Acceptance:**
- The first assistant subtitle appears quickly after LLM text starts.
- Long Chinese answers are segmented into natural spoken chunks.
- Interrupted or superseded speech chunks do not continue playing from an old turn.
- Melo sidecar failures are surfaced as degraded state rather than silent failure.

### Phase 05 — Active Listening and Partial Understanding

**Goal:** Make Blink feel present while the user is speaking, not only after the final transcript.

**Build:**
- Surface VAD and STT progress in the browser state: speech_started, speech_continuing, speech_stopped, transcribing, final_transcript.
- Show partial transcript if the STT backend provides partials; otherwise show speech activity and then final transcript without pretending partials exist.
- Add lightweight topic/constraint extraction after final transcript and optionally during long turns when reliable.
- Use active listening visually first. Avoid frequent audible backchannels until measured as helpful.
- Emit user-turn summary events for debugging and evaluation.

**Acceptance:**
- Long user turns no longer look like silence in the UI.
- The final transcript appears reliably and is clearly separated from assistant subtitles.
- Detected constraints or topics are short, editable/inspectable, and not presented as private inference certainty.
- No hallucinated partial transcripts are shown when the backend does not provide them.

### Phase 06 — Camera Presence and Grounded Browser Vision

**Goal:** Make camera-on meaningful and visible in the browser path without overusing continuous perception.

**Build:**
- Use existing LatestCameraFrameBuffer, CameraFeedHealthManager, and fetch_user_image as the baseline camera/vision path.
- Emit events when camera connects, frame received, frame stale, vision tool starts, vision returns, and vision fails.
- Show camera state, latest frame age, and whether the current answer used vision.
- Keep continuous perception off by default for daily stability unless explicitly enabled.
- Require grounded wording: if Blink used one frame, it should not claim continuous video understanding.

**Acceptance:**
- The UI clearly shows camera available, unavailable, stale, or actively looking.
- When the user asks “你看到什么/我手里拿着什么”, Blink calls fetch_user_image instead of saying it cannot see.
- Vision errors produce actionable Chinese messages.
- No camera claims are made when no fresh frame or vision result exists.

### Phase 07 — Memory and Persona Performance Compiler

**Goal:** Make memory and personality visible through behavior, not just hidden prompts.

**Build:**
- Create a PerformancePlan compiler that combines persona defaults, user memory, active project context, modality, camera state, and current turn state.
- Create a MemoryUseTrace object for each assistant turn: selected memories, suppressed memories, reason, behavior effect.
- Add a multi-reference persona library: interruption response, camera use, memory callback, disagreement, correction, concise answer, deep technical planning, uncertainty.
- Show “Used in this reply” in the browser UI with short summaries, not raw memory internals.
- Support voice commands such as “记住…”, “忘掉这个”, “纠正一下…”, and “你现在记得什么”.

**Acceptance:**
- Memory-selected responses visibly differ from memory-blind responses in relevant situations.
- The browser UI shows why a memory/persona behavior affected the reply.
- Corrected or suppressed memories stop affecting future turns.
- The persona remains coherent without pretending to be a human with a fake biography.

### Phase 08 — Blink Browser Perf Bench

**Goal:** Create a repeatable evaluation harness for both primary
browser/WebRTC paths, with native isolation guardrails included as regression
fixtures.

**Build:**
- Define scripted cases for connection, listening, speaking, camera,
  interruption, Melo/Kokoro latency, memory, persona, recovery, and native
  isolation guardrails.
- Record event traces and performance-state snapshots during sessions.
- Add automated checks for event order, stale state, missing subtitles, stale camera claims, and wrong barge-in defaults.
- Add human rating forms for state clarity, felt-heard, personality, memory usefulness, camera grounding, interruption naturalness, and enjoyment.
- Support pairwise comparisons between builds, borrowing the LPM-Bench spirit of functional and holistic evaluation.

**Acceptance:**
- A single command can run fixture validation and output a JSONL trace summary.
- Regression cases catch the native self-interruption mistake if it reappears in
  browser or native defaults.
- Human dogfooding can compare two builds with the same prompts and rating form.
- The evaluation result can answer whether the system feels more alive, not only whether it returned correct text.

### Phase 09 — Native Isolation Lane and Regression Guardrails

**Goal:** Keep native English Kokoro useful for backend isolation while preventing it from being mistaken for the primary product path.

**Build:**
- Document native English Kokoro as mic→STT→LLM→TTS backend isolation, not daily voice+camera UX.
- Preserve protected playback default in native scripts.
- Add tests asserting native barge-in is off by default and only enabled with explicit --allow-barge-in or environment variable.
- Make native camera wrapper status unambiguous: if camera path is disabled or isolation-only, say so clearly.
- Ensure any future change to barge-in policy must update docs and regression tests.

**Acceptance:**
- Native path never silently advertises a product-grade camera/interruption experience.
- Default native run lines include barge_in=off and tts=kokoro when applicable.
- Regression tests prevent forced barge-in on laptop-speaker setups.
- Browser/WebRTC remains the recommended daily interactive surface through the
  equal primary `browser-zh-melo` and `browser-en-kokoro` paths.

## Phase dependency logic

The first dependency is observability. Do not tune personality or memory before the browser can show whether Blink is listening, thinking, speaking, looking, interrupted, or waiting.

The second dependency is safe turn-taking. Do not make barge-in the default until echo health and stale-output cancellation are measurable. The native PyAudio regression proves that an implementation can appear interruptible while actually self-interrupting.

The third dependency is performable speech. Melo output must be chunked, subtitle-aligned, and cancellation-safe before long-form persona work will feel good.

The fourth dependency is visible memory/persona. The assistant should not merely have memory; the user should be able to see when memory changed the reply.

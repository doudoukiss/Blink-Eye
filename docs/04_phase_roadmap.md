# 04 — Multiphase roadmap


## Phase 00 — Reality audit and browser/Melo baseline lock

**Goal:** Establish the current product path as the stable baseline and prevent native/PyAudio assumptions from leaking back into UX work.

### Build
- Document the canonical daily path: ./scripts/run-local-browser-melo.sh -> http://127.0.0.1:7860/client/.
- Record the runtime profile, TTS backend label, WebRTC browser media mode, vision defaults, protected playback defaults, and known native Kokoro limitations.
- Add a baseline smoke checklist and a generated run manifest under artifacts/runtime_traces/.
- Add regression notes that native English Kokoro is backend isolation only, not the primary UX path.

### Acceptance
- A new developer can identify the primary UX path in under one minute.
- Smoke run prints browser/Melo, WebRTC, camera, protected playback, and log-file status clearly.
- Docs explicitly warn that native --allow-barge-in requires headphones or echo-safe setup.
- No code path silently promotes native Kokoro over browser/Melo for daily testing.

### Current and Target Files
- docs/bilingual_actor_runtime/README.md
- docs/bilingual_actor_runtime/01_dual_path_baseline.md
- docs/debugging.md
- scripts/evals/run-bilingual-actor-bench.py
- tests/test_local_workflows.py


## Phase 01 — Actor event ledger and replayable session trace

**Goal:** Upgrade the existing performance event bus into a durable, replayable actor timeline that can drive UI, debugging, evals, and future avatar adapters.

### Build
- Define a public-safe actor event vocabulary for listening, thinking, speaking, looking, interruption, memory/persona, camera, and degradation.
- Persist bounded JSONL traces per session without raw audio, raw image data, secrets, SDP, or full sensitive text unless explicitly allowed for local debug.
- Add /api/runtime/actor-events and /api/runtime/actor-trace endpoints or extend existing performance-events with schema_version=2.
- Add a replay helper that can render a timeline from a saved trace without running live media.

### Acceptance
- Every user turn produces a complete event sequence from speech start to waiting/speaking/error.
- Trace payload passes a sanitizer test that rejects raw audio/image/SDP/token/prompt fields.
- Replay tool reconstructs mode transitions and key metrics from JSONL.
- Existing /api/runtime/performance-events clients continue to work.

### Current and Target Files
- src/blink/interaction/actor_events.py
- src/blink/interaction/performance_events.py
- src/blink/cli/local_browser.py
- scripts/evals/replay-actor-trace.py
- tests/test_actor_event_ledger.py


## Phase 02 — Conversation floor controller

**Goal:** Create an explicit full-duplex conversation-floor model so Blink knows who has the floor, when overlap is benign, and when it must yield.

### Build
- Add a deterministic floor state model: user_has_floor, assistant_has_floor, overlap, handoff, repair, idle.
- Consume VAD, partial/final STT, TTS start/stop, interruption frames, assistant chunk state, and WebRTC media health.
- Classify user speech during assistant output as backchannel, accidental echo, intentional barge-in, or unclear overlap.
- Expose floor state in /api/runtime/performance-state and the UI panel.

### Acceptance
- Floor state changes are visible in the browser status panel.
- Backchannels such as “嗯”, “okay”, “right” do not always kill the assistant response.
- Clear interruptions emit overlap -> repair/yield transitions.
- The controller never enables unsafe speaker-mode barge-in without echo-safe policy approval.

### Current and Target Files
- src/blink/interaction/floor_control.py
- src/blink/interaction/barge_in.py
- src/blink/interaction/browser_state.py
- web/client_src/src/assets/blink-expression-panel.js
- tests/test_browser_floor_control.py


## Phase 03 — Adaptive interruption and echo-health policy

**Goal:** Make browser/WebRTC interruption feel human while preserving protected playback defaults and avoiding self-interruption regressions.

### Build
- Add echo-health telemetry normalization: AEC, noise suppression, AGC, audio output route if available, headphones hint, protected/barge-in status.
- Implement an adaptive interruption gate with policy outputs: protected, armed, suppressed_echo_risk, accepted, rejected_backchannel, rejected_noise.
- Add stale Melo audio chunk invalidation using generation tokens after accepted interruption.
- Add interruption metrics: speech-to-stop latency, accept/reject counts, false-positive tags, stale-drop count, resume-listening latency.

### Acceptance
- Speaker mode remains protected by default.
- Headphone/echo-safe mode can enable barge-in and stop speech quickly on clear interruption.
- The UI displays whether interruption was accepted, rejected, or protected.
- Regression cases cover self-interruption, cough, backchannel, real interruption, and sidecar TTS backlog.

### Current and Target Files
- src/blink/interaction/adaptive_interruption.py
- src/blink/interaction/barge_in.py
- src/blink/brain/speech_director.py
- src/blink/cli/local_browser.py
- evals/browser_actor_bench/regression_interruption_echo.jsonl
- tests/test_adaptive_interruption.py


## Phase 04 — Melo speech performance director v2

**Goal:** Move from text-to-TTS to performable speech: short chunks, subtitle-before-audio, controlled lookahead, boundary-aligned updates, and cancellation safety.

### Build
- Define SpeechPerformanceChunk with text, role, interruptibility, pause_after_ms, display policy, generation token, and latency metrics.
- Split assistant responses into voice-suitable chunks with sentence boundaries and Chinese punctuation support.
- Keep Melo queue lookahead small to reduce interruption latency while avoiding choppy playback.
- Emit subtitle events before audio, audio-start events when first PCM is observed, and speech-done at final chunk.

### Acceptance
- Long answers are chunked into speakable units rather than one large uninterruptible blob.
- Subtitles show what Blink is saying before or as audio starts.
- Accepted interruption invalidates old chunks and prevents stale audio from speaking later.
- Metrics include first_subtitle_latency_ms, first_audio_latency_ms, queue_depth, stale_chunk_drop_count.

### Current and Target Files
- src/blink/brain/speech_director.py
- src/blink/brain/processors.py
- src/blink/services/local_http_wav/tts.py
- src/blink/interaction/speech_performance.py
- tests/test_speech_performance_director.py


## Phase 05 — Active listener v2: understanding while silent

**Goal:** Make Blink visibly present while the user speaks by showing receipt, partial understanding, constraints, uncertainty, and readiness without talking over the user.

### Build
- Extend active-listening snapshots with heard_text_state, semantic_hints, uncertainty, ready_to_answer, and repair_needed flags.
- Support partial transcript hints when available; fall back to final-transcript deterministic extraction when not available.
- Detect project constraints, topic shifts, requests, corrections, memory commands, and camera-intent utterances.
- Add UI affordances: listening duration, live hint pills, “enough to answer” state, and “still listening” state.

### Acceptance
- A 60-second user turn no longer feels like speaking into a void.
- The panel shows at least topics/constraints/camera-intent after final transcript and partial hints when available.
- No audible backchannels are added by default.
- Hints are bounded, editable-looking, and safe; no raw long transcript is persisted unless local debug mode allows it.

### Current and Target Files
- src/blink/interaction/active_listening.py
- src/blink/interaction/floor_control.py
- src/blink/cli/local_browser.py
- web/client_src/src/assets/blink-expression-panel.js
- tests/test_active_listener_v2.py


## Phase 06 — Camera scene state and grounded browser perception

**Goal:** Turn camera-on into a trustworthy perception channel: clear availability, frame freshness, on-demand vision use, and low-rate social/scene signals without pretending continuous understanding.

### Build
- Define CameraSceneState: available, stale, user_present, face_visible, object_shown_candidate, lighting_quality, latest_frame_seq, latest_frame_age_ms, last_used_for_answer.
- Keep Moondream/vision analysis on-demand by default; add optional low-rate scene-signal extraction only when enabled.
- Tag every vision-grounded answer with frame sequence and frame age.
- Expose camera state and last-vision grounding in UI and actor traces.

### Acceptance
- Blink never claims continuous scene understanding when only a single frame was analyzed.
- Camera permission, freshness, and vision-use state are always visible.
- Visual answers include a traceable frame seq/age in the performance state.
- Camera failures degrade gracefully with actionable status.

### Current and Target Files
- src/blink/interaction/camera_presence.py
- src/blink/brain/perception/
- src/blink/cli/local_browser.py
- web/client_src/src/assets/blink-expression-panel.js
- tests/test_camera_scene_state.py


## Phase 07 — Persona reference bank and performance compiler

**Goal:** Replace prose-only personality with multi-reference behavioral conditioning: retrieve relevant persona examples and compile them into turn-level performance plans.

### Build
- Create a persona reference bank with examples for interruption, disagreement, correction, deep technical planning, visual grounding, uncertainty, memory recall, and concise casual chat.
- Compile PersonaInteractionPlan from persona defaults + memory + modality + camera state + floor state + user intent.
- Attach plan summaries and behavior effects to the current memory/persona performance payload.
- Make the LLM prompt receive compact behavior directives rather than a long hidden personality essay.

### Acceptance
- Two similar turns in different floor/camera/memory contexts produce appropriately different behavior.
- The UI can show which persona references influenced the current reply.
- The system does not invent fake human biography or emotions.
- Persona behavior is stable across interruption, correction, camera use, and technical planning.

### Current and Target Files
- src/blink/brain/persona/reference_bank.py
- src/blink/brain/persona/compiler.py
- src/blink/brain/persona/performance_plan.py
- src/blink/brain/runtime.py
- tests/test_persona_performance_compiler.py


## Phase 08 — Memory continuity and relationship model

**Goal:** Make memory visible, editable, and consequential across sessions so the user feels continuity rather than a hidden database.

### Build
- Add MemoryContinuityTrace per reply: selected memories, suppressed memories, conflicts, corrections, behavior effects, and freshness.
- Add voice/browser commands: remember, correct, forget, mark stale, what do you remember about this project?
- Improve session-to-session continuity around project constraints, user preferences, relationship style, and teaching style.
- Add UI controls for pin/suppress/correct/forget and display “used in this reply” records.

### Acceptance
- A stored project constraint changes a later answer and the UI shows the effect.
- Stale/conflicting memories are not blindly repeated.
- User corrections update or deprecate memory rather than adding contradiction.
- Memory trace remains bounded and privacy-aware.

### Current and Target Files
- src/blink/brain/memory_v2/use_trace.py
- src/blink/brain/memory_v2/governance_actions.py
- src/blink/brain/context/compiler.py
- src/blink/cli/local_browser.py
- web/client_src/src/assets/blink-operator-workbench.js
- tests/test_memory_continuity_trace.py


## Phase 09 — Browser actor surface: from debug panel to living interaction UI

**Goal:** Turn the browser client into an app-like actor surface that communicates attention, speech, thinking, camera use, and memory without overwhelming the user.

### Build
- Add a primary actor status area separate from advanced debug details.
- Show five stable regions: current mode, heard text, Blink is saying, camera/vision, memory/persona used.
- Add a compact timeline strip using actor events: listen -> heard -> think -> look -> speak -> wait.
- Use an abstract/non-human presence indicator instead of a realistic human avatar for this stage.

### Acceptance
- A non-technical tester can always tell whether Blink is listening, thinking, speaking, looking, interrupted, or waiting.
- Advanced metrics remain available but do not dominate the main UX.
- The UI does not require a full frontend rewrite; it works with the existing asset-copy build system.
- No fonts or external binary assets are introduced into the source tree.

### Current and Target Files
- web/client_src/src/assets/blink-expression-panel.js
- web/client_src/src/index.html
- generated browser package assets, when rebuilt locally
- tests/test_browser_actor_surface_payloads.py


## Phase 10 — Blink-Actor-Bench and release gate

**Goal:** Measure whether Blink is actually becoming more enjoyable, coherent, interruptible, and person-like in observable behavior.

### Build
- Create eval suites for listening, speaking, interruption, camera grounding, memory continuity, persona consistency, repair, and long-session stability.
- Support G/S/B pairwise comparison and 1–5 Likert scoring, inspired by LPM-Bench but adapted to browser voice agents.
- Add automated trace validators for missing events, stale mode, unsafe barge-in, camera false-claims, and memory contradictions.
- Add release gate thresholds for state clarity, interruption latency, memory usefulness, and dogfooding enjoyment.

### Acceptance
- Each browser/Melo build can be compared against a previous trace bundle.
- Automated tests fail when state transitions are missing or unsafe claims appear.
- Human rating form captures felt_heard, state_clarity, interruption_naturalness, personality, memory_usefulness, camera_grounding, enjoyment.
- Release gate can block regressions before demos.

### Current and Target Files
- src/blink/brain/evals/actor_bench.py
- evals/browser_actor_bench/*.jsonl
- scripts/evals/eval-browser-actor-bench.sh
- src/blink/brain/evals/release_gate.py
- tests/test_actor_bench_trace_validators.py


## Phase 11 — Privacy, safety, and consent controls

**Goal:** Make voice, camera, memory, and future avatar features trustworthy by design.

### Build
- Add visible disclosure for camera/mic active state, on-demand vision use, and memory writes.
- Add privacy modes: no_trace, local_trace, debug_trace; each with strict event redaction rules.
- Add consent gates for continuous perception and any future human-like avatar identity conditioning.
- Add safety tests for raw image/audio leakage, prompt leakage, memory over-retention, and non-consensual likeness claims.

### Acceptance
- The user can tell when camera and mic are active and when a frame is used for an answer.
- No trace contains raw audio/image/SDP/token by default.
- Continuous perception requires explicit opt-in and a visible UI state.
- Future avatar integrations are gated behind identity/likeness consent requirements.

### Current and Target Files
- src/blink/interaction/privacy.py
- src/blink/interaction/actor_events.py
- src/blink/cli/local_browser.py
- web/client_src/src/assets/blink-expression-panel.js
- tests/test_actor_privacy_safety.py


## Phase 12 — Avatar adapter readiness without building the avatar yet

**Goal:** Prepare the architecture for a future LPM-like or third-party avatar engine while keeping the current product lightweight and browser/Melo-first.

### Build
- Define AvatarPerformanceAdapter interface that consumes actor events, speech chunks, floor state, camera state, and persona plan.
- Create a null/abstract adapter that drives only the existing UI presence indicator.
- Add optional adapters for existing video services only as stubs/specs; do not require a realistic human avatar in the baseline.
- Define consent, watermark/disclosure, and identity-reference requirements for any future human-like avatar plugin.

### Acceptance
- The current browser/Melo path works with the null adapter only.
- Future avatar engines can subscribe to the same actor event stream without changing STT/LLM/TTS core.
- No realistic human identity generation is introduced by this phase.
- Docs explain the boundary between Blink as an AI presence and any optional rendered character.

### Current and Target Files
- src/blink/interaction/avatar_adapter.py
- src/blink/interaction/actor_events.py
- docs/bilingual_actor_runtime/11_browser_actor_surface.md
- tests/test_avatar_adapter_contract.py

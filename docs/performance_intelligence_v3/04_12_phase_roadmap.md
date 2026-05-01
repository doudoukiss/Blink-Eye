# Blink Bilingual Performance Intelligence V3: 12-phase roadmap

This roadmap has exactly 12 implementation phases. It builds on the existing bilingual actor runtime and keeps both local browser product paths equal.

## Phase 01 — Baseline lock and dual-path reality audit

**Goal:** Freeze the current bilingual actor runtime as the non-regression baseline before building the next intelligence layer.

**Build:**
- Read the current bilingual actor runtime docs, launchers, state/event schemas, tests, browser actor surface, and release-gate scripts.
- Generate a machine-readable `PerformanceIntelligenceBaseline` snapshot for both profiles: profile, language, TTS runtime label, browser vision default, continuous perception default, protected playback default, actor-state endpoints, actor-event endpoints, and current gate commands.
- Add baseline fixtures under `evals/bilingual_performance_intelligence_v3/baseline_profiles.json` for `browser-zh-melo` and `browser-en-kokoro`.
- Add one test that fails if either canonical launcher silently disables Moondream/browser vision by default or demotes a path to isolation/debug-only status.
- Document native PyAudio paths as backend isolation lanes, not product UX lanes. Do not delete or break them.

**Acceptance:**
- Both canonical browser paths are explicitly represented in the baseline snapshot with equal product status.
- The release baseline can be regenerated deterministically without opening the browser or calling models.
- Regression tests prevent future path drift: wrong profile, wrong language, wrong TTS label, vision off by default, or protected playback default removed.
- The docs clearly state that this V3 upgrade builds on the existing actor runtime rather than replacing it.

## Phase 02 — Performance episode ledger and replay v3

**Goal:** Promote actor events into privacy-safe performance episodes that can be replayed, scored, compared, and used for learning without storing raw media or hidden prompts.

**Build:**
- Create `PerformanceEpisodeV3`: a compact session/turn/subturn ledger built from actor events, floor transitions, active-listener state, speech chunks, camera-scene state, memory-use trace, persona plan, interruption decisions, degradation events, and user-visible bounded text.
- Add `scripts/evals/replay-performance-episode-v3.py` that reconstructs timeline, modes, latencies, floor state, camera use, interruption outcome, memory/persona effect, and failure labels from saved episode JSONL.
- Separate live state from persistent trace: live UI can show bounded transcript/subtitle text; persistent episode stores IDs, hashes, counts, labels, and reason codes by default.
- Add a sanitizer with denylist and shape checks for raw audio, raw images, SDP, ICE, credentials, hidden prompts, full model messages, raw memory bodies, and unbounded transcripts.
- Define episode segment types: `listen_segment`, `think_segment`, `look_segment`, `speak_segment`, `overlap_segment`, `repair_segment`, `idle_segment`.

**Acceptance:**
- A Chinese/Melo run and an English/Kokoro run can produce structurally identical episode ledgers except for profile/language/TTS labels.
- Replay works offline without audio, camera, browser, Moondream, LLM, MeloTTS, or Kokoro.
- Unsafe payload tests fail closed.
- Episode summaries can power evaluation and dogfooding review without exposing private raw media or hidden internals.

## Phase 03 — Actor control-frame scheduler with boundary-aligned updates

**Goal:** Add a state-adaptive scheduler that converts evolving user/audio/vision/memory/persona conditions into bounded control frames applied at safe interaction boundaries.

**Build:**
- Introduce `ActorControlFrameV3`, inspired by LPM's online runtime: persistent state is separated from refreshable condition caches; updates apply at explicit boundaries; lookahead is bounded to avoid backlog and stale speech.
- Define boundary types: `vad_boundary`, `stt_final_boundary`, `speech_chunk_boundary`, `tts_queue_boundary`, `camera_frame_boundary`, `tool_result_boundary`, `interruption_boundary`, and `repair_boundary`.
- Implement `ActorControlScheduler` that consumes actor events and emits control frames for browser UI, speech director, active listener, floor controller, camera scene state, persona planner, and memory compiler.
- Add controlled lookahead counters for speech chunks and UI subtitles so the assistant cannot generate a long uninterruptible backlog.
- Ensure scheduler policy is deterministic and testable; the LLM should not control low-level boundary timing.

**Acceptance:**
- Control-frame replay reconstructs state transitions deterministically.
- New user input after an interruption affects the next safe boundary and stale output is marked dropped/suppressed.
- Bounded lookahead prevents large TTS queues from masking interruption and repair.
- Both primary profiles use the same scheduler policy with language-specific labels only.

## Phase 04 — Conversation floor v3: overlap, repair, and bilingual backchannels

**Goal:** Make turn-taking more human-like by explicitly modeling overlap, continuers, repair, and yield decisions across Chinese and English.

**Build:**
- Extend the current floor controller with richer sub-states: `user_holding_floor`, `assistant_holding_floor`, `overlap_candidate`, `accepted_interrupt`, `ignored_backchannel`, `repair_requested`, `handoff_pending`, `handoff_complete`.
- Add bilingual backchannel/continuer phrase sets and phrase-class confidence: Chinese examples include `嗯`, `对`, `继续`, `等一下`, `不是`; English examples include `yeah`, `right`, `go on`, `wait`, `no`.
- Combine acoustic timing, STT partial/final content, assistant speaking state, TTS chunk role, and WebRTC echo-health state.
- Emit explicit reason codes: `short_backchannel`, `explicit_interrupt`, `correction`, `user_continuing`, `assistant_pause`, `echo_risk`, `protected_playback`, `low_confidence_transcript`.
- Add deterministic fixtures for cross-lingual interruption and repair cases.

**Acceptance:**
- Backchannels no longer kill speech unless policy says to yield.
- Explicit corrections and interruption phrases yield quickly when echo-safe or explicitly armed.
- Repair state is visible in actor state and episode replay.
- Chinese and English floor behavior are tested symmetrically.

## Phase 05 — Active Listener semantic state v3

**Goal:** Make Blink visibly understand the user while listening, without interrupting or pretending to know more than it does.

**Build:**
- Create `SemanticListenerStateV3`: current topic, detected user intent, constraints, uncertainty, emotional tone label, language, whether enough information is available to answer, and short safe live-summary text.
- Use STT partials/finals, turn duration, camera-scene hints, memory context, and floor state to update listener state incrementally.
- Expose listener chips in the browser UI: `I heard...`, `constraint detected`, `question detected`, `showing object`, `still listening`, `ready to answer`.
- Keep live summaries bounded and ephemeral; persistent episodes store labels/hashes/reason codes unless debug trace is explicitly enabled.
- Add bilingual semantic fixtures for long user turns, confused corrections, object-showing requests, and project-planning constraints.

**Acceptance:**
- Long user turns no longer feel like speaking into a void; state changes while the user speaks.
- Listener state does not create audible backchannels by default.
- Chinese and English listener chips are localized or label-compatible.
- The active listener never claims camera understanding unless a fresh frame or explicit camera-scene signal supports it.

## Phase 06 — Scene-social perception v2 with Moondream grounding

**Goal:** Upgrade camera use from a raw vision tool into an honest scene-state system that supports presence, object-showing, and grounding parity for both primary paths.

**Build:**
- Extend `CameraSceneState` into `SceneSocialStateV2`: camera permission, frame freshness, user presence, face/body/hands/object hints, object-showing likelihood, scene-change reason, last Moondream result, and confidence/age metadata.
- Keep continuous perception off by default. Support on-demand `fetch_user_image` and optional low-frequency scene-state updates only when explicitly enabled.
- Add scene-state transitions to actor control frames and performance episodes: `camera_ready`, `looking_requested`, `frame_captured`, `vision_answered`, `vision_stale`, `vision_unavailable`.
- Add camera honesty policy: Blink must distinguish `I can see now`, `I have a recent frame`, `camera is available but not used`, and `camera is unavailable`.
- Add bilingual Moondream parity tests for object questions, frame staleness, user asking whether camera is on, and no-vision fallback.

**Acceptance:**
- Both Chinese/Melo and English/Kokoro paths expose equivalent camera scene state.
- No false camera claims appear in deterministic tests or episode replay.
- Scene-state updates do not store raw images in public traces.
- The UI makes looking/grounding state visible before and after Moondream calls.

## Phase 07 — Performance planner v3 and ActorControl compiler

**Goal:** Compile memory, persona, floor state, scene state, listener state, and TTS capabilities into a single visible performance plan per turn.

**Build:**
- Define `PerformancePlanV3`: stance, response shape, voice pacing, chunk budget, subtitle policy, camera reference policy, memory callback policy, interruption policy, repair policy, UI status copy, and reason trace.
- Compile the plan before response generation and refresh it at safe boundaries when new user/camera/interruption information arrives.
- Make the plan consume `ActorControlFrameV3`, `SemanticListenerStateV3`, `SceneSocialStateV2`, `MemoryContinuityTrace`, persona references, and TTS capability declarations.
- Expose a bounded `plan_summary` in actor state and episode replay so the user and evaluator can see why Blink is acting a certain way.
- Do not add fake human biography or unsupported emotions. The plan expresses an AI persona through behavior, pacing, memory, and repair.

**Acceptance:**
- Every assistant turn has a plan summary and reason trace.
- The plan changes behavior observably: shorter voice chunks, camera acknowledgement, memory callback, repair stance, or uncertainty handling.
- Chinese and English plans differ only where language/TTS/copy requires it.
- Tests catch unsupported claims such as emotional prosody when the TTS backend does not expose that control.

## Phase 08 — Dual TTS speech director v3 for MeloTTS and Kokoro

**Goal:** Make voice output sound directed and easy to interrupt while respecting each backend's real capabilities.

**Build:**
- Extend `SpeechPerformanceChunk` with role, language, estimated duration, display text, interruptibility, pause-after, subtitle timing, stale-generation token, and backend-specific capability flags.
- For MeloTTS, preserve Chinese text normalization and sidecar health behavior. For Kokoro, preserve English voice path and avoid assumptions about emotional prosody or arbitrary controls unless actually implemented.
- Add chunk budgeting: voice mode should avoid huge monologues and should prefer semantically complete chunks aligned with interruption boundaries.
- Emit subtitles before or at playback start, not after audio finishes.
- Integrate stale-output dropping with floor/interruption state and actor control-frame scheduler.

**Acceptance:**
- A long technical answer is split into performable chunks with visible subtitles.
- Accepted interruption prevents old chunks from playing later.
- Melo and Kokoro produce structurally similar speech chunks with backend-specific capability labels.
- The system does not claim backend controls that are not supported by the integration.

## Phase 09 — Persona multi-reference bank and cross-language expression anchors

**Goal:** Convert personality from hidden prose into a stable set of situational behavior references that work in both Chinese and English.

**Build:**
- Create or extend `PersonaReferenceBankV3` with multi-reference persona anchors analogous to LPM identity references: interruption response, correction response, deep technical planning, casual check-in, visual grounding, uncertainty, disagreement, memory callback, and playful-but-not-fake-human behavior.
- Each reference must include Chinese and English examples or language-neutral behavior constraints, plus negative examples to avoid.
- Add retrieval by situation, not by generic persona keywords. The current performance plan should cite which references influenced the turn.
- Add cross-language consistency tests: same situation in Chinese and English should produce recognizably same Blink stance while respecting language norms.
- Allow user-visible inspection of style anchors without exposing hidden prompts.

**Acceptance:**
- Persona is observable through response shape, timing, memory use, repair, and camera honesty rather than only a system prompt.
- Chinese and English persona behavior is coherent, not two unrelated assistants.
- Negative examples prevent fake-human intimacy, invented biography, and repetitive catchphrases.
- Persona reference selection appears in plan summaries and evaluation traces.

## Phase 10 — Memory continuity v3 and discourse episode model

**Goal:** Make long-horizon continuity felt through relevant callbacks, corrections, and cross-language relationship memory, not through indiscriminate prompt stuffing.

**Build:**
- Add `DiscourseEpisode` records derived from performance episodes: active project, user preferences, unresolved commitments, corrections, visual events, repeated frustrations, and successful interaction patterns.
- Link memory retrieval to performance planning: selected memory must have an explicit behavioral effect such as shorter explanation, recall of current project constraints, corrected preference, or avoided repetition.
- Add cross-language memory normalization: a preference taught in Chinese should affect English interaction and vice versa, with language-appropriate phrasing.
- Add memory conflict and stale-memory handling: newer corrections supersede older claims; uncertain memories are presented as tentative or suppressed.
- Expose a bounded memory-use trace in UI and episodes: memory ID/summary/effect/reason, not raw private memory bodies.

**Acceptance:**
- Memory use visibly changes the next turn when relevant.
- Cross-language preference transfer works in deterministic fixtures.
- Stale or contradicted memories are suppressed or corrected.
- The assistant can answer 'why did you respond that way?' with a concise public-safe plan/memory trace.

## Phase 11 — Performance learning flywheel and browser workbench

**Goal:** Turn dogfooding into structured preference data so the performance planner improves rather than relying on impressions.

**Build:**
- Add a browser/developer workbench panel for episode replay, actor state timeline, performance plan summaries, floor decisions, camera-state claims, memory/persona trace, and rating controls.
- Create `PerformancePreferencePair`: two candidate plans/responses or two build traces, rated on felt-heard, state clarity, interruption naturalness, voice pacing, camera honesty, memory usefulness, persona consistency, enjoyment, and not-fake-human.
- Implement DPO-lite policy iteration without model fine-tuning first: use preferences to update prompt references, chunking policy, memory callback thresholds, and interruption/backchannel rules.
- Add dogfooding scripts for 5-minute Chinese and English conversations with specific tasks, visual questions, interruptions, corrections, and memory callbacks.
- Keep human ratings local and privacy-safe by default; no raw media export unless explicitly enabled.

**Acceptance:**
- Dogfooding produces structured preference JSONL, not free-form notes only.
- A release candidate can be compared against baseline with pairwise outcomes.
- Policy changes must cite the preference evidence that motivated them.
- The workbench does not expose hidden prompts, raw camera frames, raw audio, or secrets by default.

## Phase 12 — Bilingual Performance Bench v3, release gate, and avatar-adapter contract

**Goal:** Expand the release gate from actor-state correctness to perceived performance quality, privacy, parity, and future avatar readiness.

**Build:**
- Create `BilingualPerformanceBenchV3` with matched Chinese and English cases for connection, listening, speech, overlap/interruption, camera grounding, repair, memory/persona, long-session continuity, preference comparison, and safety controls.
- Add hard blockers: profile regression, hidden camera use, false camera claim, self-interruption, stale TTS after interrupt, memory contradiction, unsupported TTS claim, unsafe trace payload, missing consent controls, and realistic-human avatar capability.
- Add quantitative metrics: state clarity, perceived responsiveness proxy, interruption stop latency, stale chunk drops, camera frame age policy, memory-effect rate, persona-reference hit rate, episode sanitizer pass rate, and bilingual parity delta.
- Extend avatar-adapter contract to accept only public-safe actor events/control frames/plan summaries. It may drive abstract/status/symbolic avatars, not realistic human likeness, identity cloning, face reenactment, raw media, or hidden prompts.
- Document release process: run profile gates, browser workflow tests, episode replay, sanitizer, preference review, and manual dogfooding before merge.

**Acceptance:**
- The V3 bench runs both canonical paths and fails if either path falls below threshold or lacks Moondream parity.
- All hard blockers are executable checks or explicit human-review gate items.
- Avatar contract remains safe and media-free by default.
- A release report explains whether this build is more alive, not merely more instrumented.

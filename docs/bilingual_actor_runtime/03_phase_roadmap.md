# 12-phase roadmap

This roadmap contains exactly 12 implementation phases for preserving parity
between the Chinese MeloTTS and English Kokoro browser product paths while the
actor runtime grows.

## Phase 01 — Dual-path baseline parity and launchers

**Goal:** Make Chinese browser/MeloTTS/Moondream and English browser/Kokoro/Moondream equal first-class local product paths.

**Build:**
- Treat `./scripts/run-local-browser-melo.sh` as the canonical Chinese conversation path: profile `browser-zh-melo`, TTS `local-http-wav/MeloTTS`, language Chinese, browser/WebRTC media, Moondream/browser vision on by default, continuous perception off by default.
- Treat `./scripts/run-local-browser-kokoro-en.sh` as the canonical English conversation path: profile `browser-en-kokoro`, TTS `kokoro/English`, language English, browser/WebRTC media, Moondream/browser vision on by default, continuous perception off by default.
- Update the English launcher if it still sets `BLINK_LOCAL_BROWSER_VISION=0`; equal path status requires browser vision/Moondream availability unless explicitly disabled by the user.
- Add clear startup banners for both launchers: profile, language, TTS runtime label, WebRTC, camera/vision status, protected playback, barge-in policy, log file, and client URL.
- Add a baseline compatibility document explaining that native PyAudio paths are backend isolation paths, not primary daily UX paths.

**Acceptance:**
- A developer can run either canonical script and immediately see the active profile, language, TTS backend, vision status, protected playback state, and log location.
- Both paths route to the same browser client URL and expose the same runtime-state schema.
- English Kokoro no longer has camera off by default in the equal-primary local browser path unless a user passes an explicit no-vision flag or environment variable.
- Regression tests prevent a future change from silently demoting either path.

## Phase 02 — Actor event schema v2 and bilingual event ledger

**Goal:** Upgrade the current browser performance events into a bilingual, public-safe actor-event ledger that supports live UI, debugging, replay, and evaluation.

**Build:**
- Define `ActorEventV2` with profile, language, TTS backend, vision backend, event type, high-level mode, session/client IDs, timestamp, safe metadata, and reason codes.
- Cover modes and events for connected, listening, speech_started, partial_heard, final_heard, thinking, looking, speaking, interrupted, waiting, error, memory_used, persona_plan_compiled, degraded, and recovered.
- Persist bounded JSONL traces under `artifacts/actor_traces/` when tracing is enabled. Default trace must not include raw audio, raw images, SDP, ICE candidates, secrets, full prompts, model messages, or unbounded transcript text.
- Keep backward compatibility with existing `BrowserPerformanceEventBus` clients; wrap or adapt instead of deleting the existing API abruptly.
- Add a replay helper that reconstructs mode transitions and metrics from a saved trace.

**Acceptance:**
- A Chinese/Melo run and an English/Kokoro run produce structurally identical actor-event traces except for profile/language/TTS labels.
- The sanitizer rejects unsafe keys and unsafe value tokens in persistent traces.
- Replay reconstructs the interaction timeline without live audio, camera, or model calls.
- Existing tests for browser performance events still pass.

## Phase 03 — Browser actor state parity API

**Goal:** Expose one browser actor-state snapshot that works identically for both Chinese/Melo/Moondream and English/Kokoro/Moondream.

**Build:**
- Create or extend `BrowserActorStateV2` as the single public-safe snapshot for the browser client and diagnostics.
- Include runtime profile, language, TTS label, WebRTC media state, microphone state, camera state, vision state, protected playback, interruption state, active-listening state, speech state, memory/persona state, degradation state, and last actor event.
- Expose state through stable endpoints such as `/api/runtime/actor-state` and keep existing endpoints working as aliases or adapters.
- Represent live UI text separately from persistent trace: live state may show bounded user-visible transcript/subtitle strings; persistent trace should store counts/hashes or be opt-in debug only.
- Add parity tests for both profiles.

**Acceptance:**
- The same client code can render Chinese/Melo and English/Kokoro states without profile-specific branching except labels and localized copy.
- State freshness and last-event IDs advance during connection, listening, thinking, looking, speaking, and waiting transitions.
- No state payload exposes raw audio, raw image, SDP, token, credential, or full internal prompt/message content.
- A schema fixture validates both profile examples.

## Phase 04 — Conversation floor controller v2

**Goal:** Make turn-taking explicit instead of leaving it implicit in VAD/STT/TTS side effects.

**Build:**
- Add a `ConversationFloorState` model with `user_has_floor`, `assistant_has_floor`, `overlap`, `handoff`, `repair`, and `unknown` states.
- Use browser/WebRTC VAD events, STT interim/final events, TTS start/stop events, LLM start/end events, and interruption decisions as inputs.
- Support Chinese and English backchannels, short confirmations, hesitation, correction, and explicit interruption phrases.
- Emit actor events for floor transitions and attach reason codes explaining why the controller yielded, continued, waited, or repaired.
- Keep the controller deterministic and unit-testable; do not rely on the LLM for low-level floor physics.

**Acceptance:**
- The floor controller can distinguish user turn start, user continuing, short pause, final user turn, assistant turn, overlap, and repair in deterministic fixtures.
- Backchannels such as `嗯`, `对`, `ok`, `yeah`, and `right` do not automatically terminate assistant speech when policy says continue.
- Explicit interruption phrases trigger overlap/yield policy when barge-in is armed or echo-safe.
- Floor state appears in actor state and actor events for both profiles.

## Phase 05 — Adaptive interruption and WebRTC echo health

**Goal:** Use browser/WebRTC media advantages while preventing self-interruption, false interruptions, and unsafe speaker-mode assumptions.

**Build:**
- Keep protected playback as the default unless the controller determines the session is echo-safe or the user explicitly arms barge-in.
- Add a WebRTC audio-health snapshot covering mic readiness, track state, output playback state, available stats, echo-risk hints, and whether barge-in is protected/armed/adaptive.
- Integrate floor-controller output with interruption policy: backchannel, explicit interruption, acoustic noise, cough, transcript confidence, and assistant-speech state.
- Add stale-output guards that prevent old TTS chunks from playing after accepted interruption.
- Emit actor events for interruption_candidate, interruption_accepted, interruption_rejected, output_flushed, and interruption_recovered.

**Acceptance:**
- Speaker-mode default cannot self-interrupt after a word or two.
- Headphone/echo-safe mode can interrupt reliably, with visible confirmation in UI state.
- False interruption cases are classified and counted rather than hidden.
- Both Chinese/Melo and English/Kokoro share the same interruption policy with language-specific phrase lists.

## Phase 06 — Dual TTS speech performance director for MeloTTS and Kokoro

**Goal:** Make both TTS paths perform speech as interruptible, subtitle-ready chunks rather than opaque monologues.

**Build:**
- Define `SpeechPerformanceChunk` with role, text, language, backend, display text, pause-after hint, interruptibility, context ID, and generation token.
- Implement or extend `BrainExpressionVoicePolicyProcessor` / `speech_director.py` so Chinese/Melo and English/Kokoro use backend-specific chunking while sharing one public contract.
- Emit subtitles before or at TTS enqueue time so the user knows what Blink is saying even if audio is delayed.
- Bound the TTS backlog; prefer short performable chunks in voice mode and drop stale chunks after accepted interruption.
- Maintain a backend capability map. Do not claim emotional prosody controls unless the backend integration actually supports them.

**Acceptance:**
- Long answers are chunked into voice-suitable units in both Chinese and English.
- The browser UI can show current-speech subtitles before audio playback completes.
- Accepted interruption prevents old Melo/Kokoro chunks from continuing to play.
- Backend capability tests document what Melo and Kokoro can and cannot do.

## Phase 07 — Active Listener v2: bilingual understanding while the user speaks

**Goal:** Make Blink visibly listen and extract lightweight, editable understanding cues before the final answer.

**Build:**
- Extend active-listening snapshots with bilingual phases, interim/final transcript availability, topic hints, constraint hints, uncertainty flags, and readiness-to-answer state.
- Support Chinese and English heuristics for constraints, topic nouns/phrases, correction phrases, and project references.
- Show active listening visually first; avoid frequent audible backchannels until evaluation proves they help.
- Emit actor events for listening_started, partial_understanding_updated, final_understanding_ready, and listening_degraded.
- Make all hints bounded, editable, and safe for display. Avoid exposing private prompt/memory internals.

**Acceptance:**
- During long user turns, UI state changes from listening to speech_continuing and updates safe hint counts/labels.
- Both Chinese and English fixtures extract useful topics and constraints without raw overcollection.
- Users can tell Blink heard the main constraint before Blink speaks.
- Active-listening degradation is visible when STT or media state is unreliable.

## Phase 08 — Moondream camera scene state and grounded perception parity

**Goal:** Make camera-on meaningful and honest in both local browser paths without pretending continuous vision is always happening.

**Build:**
- Enable browser camera/Moondream support for both `browser-zh-melo` and `browser-en-kokoro` profiles by default, with explicit opt-out flags.
- Create `CameraSceneState` with camera permission/track state, latest frame sequence, frame age, freshness, on-demand vision state, last-used frame metadata, grounding mode, and degradation reason.
- Mark the assistant as `looking` when `fetch_user_image` or equivalent Moondream tool use starts, and record whether the current answer actually used vision.
- Keep continuous perception off by default; support low-frequency scene-state updates only behind explicit config.
- Add UI-ready camera states: disabled, permission_needed, waiting_for_frame, available, looking, stale, stalled, error.

**Acceptance:**
- Both primary profiles show camera availability and latest-frame freshness in actor state.
- The assistant does not claim it saw the scene unless a fresh frame was actually used or the state explicitly says it is using stale/limited visual context.
- Moondream errors degrade gracefully and produce visible reason codes.
- Tests cover Chinese and English visual questions with the same camera state contract.

## Phase 09 — Persona reference bank and performance compiler v2

**Goal:** Convert Blink’s personality from hidden prose into observable conversation moves that work in Chinese and English.

**Build:**
- Create a persona reference bank with examples for interruption, correction, disagreement, deep technical planning, casual chat, camera use, memory callback, uncertainty, concise answer, and playful restraint.
- Represent references as structured data: locale, scenario, stance, response shape, forbidden moves, example input, example output, and performance notes.
- Compile a `PerformancePlanV2` per assistant turn from profile, language, memory trace, active-listening hints, camera state, floor state, user affect/intent if available, and persona references.
- The compiler should output stance, response shape, memory callback policy, camera-reference policy, interruption policy, speech-chunking hints, and UI-state hints.
- Avoid fake human biography. Blink can have continuity, memory, style, and presence without pretending to be a human person.

**Acceptance:**
- Both Chinese and English answers can use equivalent persona references without sounding like direct translations of each other.
- The performance plan visibly changes response behavior: structure, depth, memory callback, camera wording, and repair style.
- Tests demonstrate persona consistency under repeated questions, correction, interruption, and camera use.
- The UI can show a compact `style/persona used this turn` summary.

## Phase 10 — Memory continuity and cross-language relationship model

**Goal:** Make memory feel real by making it selected, visible, editable, and behaviorally consequential across Chinese and English sessions.

**Build:**
- Add a per-turn `MemoryContinuityTrace` that records selected memory summaries, suppressed memory reasons, memory effect, language/profile, and whether the user can inspect/edit the memory.
- Support voice/text commands in both Chinese and English: remember, forget, correct, what do you remember, why did you answer that way.
- Allow cross-language retrieval where appropriate: a Chinese project preference can inform an English answer and vice versa, with clear summary translation rather than raw hidden context dumping.
- Connect memory trace to the performance compiler so selected memory changes response shape or callback policy.
- Keep privacy boundaries: no raw memory bodies in default public traces; use summaries, IDs only if safe, and user-facing edit controls.

**Acceptance:**
- Seeded memories affect later replies in both languages.
- Corrections update or suppress stale memories instead of creating contradictions.
- The UI can show `Used in this reply` with bounded human-readable summaries.
- The assistant can answer why it responded that way using memory/persona traces without revealing hidden prompts.

## Phase 11 — Browser Actor Surface v2

**Goal:** Turn the actor runtime into a clear product surface: users should always know what Blink is doing and why.

**Build:**
- Upgrade the browser UI to show a compact actor-status surface: profile badge, language, TTS, camera/Moondream, listening/thinking/looking/speaking/waiting mode, protected/armed interruption, and degradation notices.
- Show `Heard`, `Blink is saying`, `Looking`, `Used memory/persona`, and `Interruption` panels using the public actor-state API.
- Keep UI profile-neutral: Chinese/Melo and English/Kokoro should use the same components with localized labels and profile badges.
- Add a bounded timeline/debug drawer for actor events. It should be developer-useful without exposing raw private payloads.
- Use feature flags for any large visual changes; preserve the current client entrypoint `http://127.0.0.1:7860/client/`.

**Acceptance:**
- A user can tell within one glance whether Blink is listening, thinking, looking through camera, speaking, interrupted, or waiting.
- Current assistant speech text appears before or during TTS playback.
- Camera state and memory/persona use are visible in both profiles.
- The UI does not show secrets, raw SDP, raw images, or hidden prompts.

## Phase 12 — Bilingual Actor Bench, release gate, safety, and avatar-adapter readiness

**Goal:** Make quality measurable and prepare the actor-event contract for future avatar/video engines without building a realistic avatar now.

**Build:**
- Create `Blink Bilingual Actor Bench` with matched Chinese/Melo/Moondream and English/Kokoro/Moondream cases for connection, active listening, speech, camera grounding, interruption, memory/persona, recovery, and long-session stability.
- Add human rating forms for state clarity, felt-heard, voice pacing, camera grounding, memory usefulness, interruption naturalness, personality consistency, enjoyment, and not-fake-human.
- Add release gates with minimum scores and hard blockers: unsafe trace payload, hidden camera use, false camera claims, self-interruption, stale TTS after interruption, memory contradiction, profile regression.
- Add consent and privacy controls for camera, trace persistence, memory inspection/editing, and optional debug transcript storage.
- Define an `AvatarAdapterEventContract` that consumes actor events and performance plans. It should support future abstract/avatar surfaces but must not implement realistic human likeness generation in this phase.

**Acceptance:**
- One command can run deterministic bench checks for both profiles and produce a comparable JSON result.
- Release gate fails if either primary path regresses below threshold.
- Safety/privacy checks are part of the release gate, not optional documentation.
- Future avatar work can subscribe to actor events without changing turn-taking, memory, or camera logic.

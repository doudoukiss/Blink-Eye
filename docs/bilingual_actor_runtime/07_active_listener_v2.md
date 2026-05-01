# Phase 07: Active Listener v2

Active Listener v2 makes browser listening visible before Blink speaks. It is a
live UI and diagnostic surface for both primary browser paths:

- `browser-zh-melo`: Chinese + MeloTTS over `local-http-wav`
- `browser-en-kokoro`: English + Kokoro

No audible backchannels are added in this phase. The listener is visual-first
until evaluation shows spoken acknowledgements improve turn-taking.

## State Shape

`/api/runtime/actor-state` exposes `active_listening.schema_version=2`.
The state includes:

- phase: `idle`, `listening_started`, `speech_continuing`,
  `partial_understanding`, `transcribing`, `final_understanding`,
  `ready_to_answer`, `degraded`, or `error`
- transcript availability and bounded character/count metadata
- editable hint lists for topics, constraints, corrections, and project
  references
- nested `semantic_state_v3.schema_version=3` with current topic, detected
  intent label, constraint labels, uncertainty, tone label, language, readiness,
  bounded live summary, listener chips, camera reference state, memory context
  state, floor state, and reason codes
- uncertainty flags, readiness state, degradation state, timestamps, and reason
  codes

`/api/runtime/performance-state` keeps the v1-compatible active-listening shape
for existing clients.

Phase 05 keeps `/api/runtime/actor-state`, `/api/runtime/actor-events`, and the
compatibility performance endpoints stable. It adds semantic state inside the
existing actor-state payload so both primary browser paths can show localized
chips such as "I heard...", "constraint detected", "question detected",
"showing object", "still listening", and "ready to answer" without changing the
endpoint contract. The Chinese path renders compatible labels such as
"我听到...", "检测到约束", "检测到问题", "正在看物体", "继续听", and "可以回答".

## Bilingual Heuristics

The extractor is deterministic and dependency-light. It supports Chinese and
English phrase markers for:

- constraints such as "必须", "不要", "only", "avoid", and "without"
- corrections such as "更正", "改成", "actually", and "I meant"
- uncertainty such as "可能", "不确定", "maybe", and "not sure"
- project references such as Blink, MeloTTS, Kokoro, Moondream, WebRTC,
  `browser-zh-melo`, and `browser-en-kokoro`

Hints are bounded to five items per list and short labels. They are editable by
the UI and intended as lightweight understanding cues, not as a transcript.

Semantic intent labels are deterministic and public-safe: `question`,
`instruction`, `correction`, `object_showing`, `project_planning`,
`small_talk`, and `unknown`.

Camera chips never claim visual understanding from text alone. `showing_object`
requires fresh camera-scene evidence such as current vision use, fresh scene
state, or active grounding. Stale, disabled, unsupported, or error camera state
becomes `camera_limited`, `stale_or_limited`, or `error`.

## Privacy Rules

Active Listener v2 may inspect interim and final STT text in memory to derive
hints and semantic chips. The bounded `safe_live_summary` is live actor-state
text only. Public events, actor traces, default diagnostics, performance
episodes, and actor control frames persist only counts, hint kinds, semantic
labels, chip IDs, readiness/degradation state, hashes, and reason codes.

The sanitizer drops or suppresses URL/path fragments, credentials, token-like
values, hidden-prompt references, raw memory IDs, and long transcript spans.
Default payloads do not expose raw audio, raw images, SDP, ICE candidates,
secrets, hidden prompts, full messages, or unbounded transcripts.

## Actor Events

The runtime emits public-safe performance events that map into actor-event v2:

- `active_listening.listening_started` -> `listening_started`
- `active_listening.partial_understanding_updated` ->
  `partial_understanding_updated`
- `active_listening.final_understanding_ready` ->
  `final_understanding_ready`
- `active_listening.listening_degraded` -> `listening_degraded`

The event metadata is text-free and contains hint counts, hint kinds,
readiness/degradation state, semantic intent, chip IDs, summary hashes, and
bounded reason codes. Native PyAudio paths remain backend isolation lanes; the
first-class product UX paths are still the two browser/WebRTC paths.

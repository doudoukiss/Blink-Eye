# Phase 10: Memory Continuity V3 And Discourse Episode Model

Phase 10 adds a public-safe `DiscourseEpisode` model and nests
`memory_continuity_v3` inside the existing `MemoryContinuityTrace`. The v1
memory-use trace remains stable; the V3 layer is the browser/operator and
planning surface for showing why a reply used memory, which discourse episodes
were relevant, which buckets were suppressed, and whether cross-language
continuity shaped the answer.

This phase builds on the existing bilingual browser actor runtime. It does not
replace `/api/runtime/actor-state`, `/api/runtime/actor-events`,
`/api/runtime/memory`, `/api/runtime/operator`, or compatibility performance
endpoints. Native PyAudio paths remain backend isolation lanes, while
`browser-zh-melo` and `browser-en-kokoro` remain equal primary product lanes.

## Public Payload

`MemoryContinuityTrace` lives under `src/blink/brain/memory_v2/continuity_trace.py`
and validates against `schemas/memory_continuity_trace.schema.json`.
`DiscourseEpisode` lives under `src/blink/brain/memory_v2/discourse_episode.py`
and validates against `schemas/discourse_episode_v3.schema.json`.

The trace includes:

- runtime profile, language, turn/session/user/thread IDs, and timestamp;
- selected memory summaries, display kinds, safe scoped IDs, provenance labels,
  editability flags, and cross-language flags;
- suppressed memory buckets such as `suppressed`, `historical`, and `limit`;
- `memory_effect`: `none`, `callback_available`, `cross_language_callback`, or
  `repair_or_uncertainty`;
- nested `memory_continuity_v3` with selected discourse episode IDs, category
  labels, memory effect labels, conflict/staleness labels, cross-language
  transfer count, and public-safe reason codes;
- bounded command intent for remember, forget, correct, list memory, and explain
  answer requests.

Discourse episodes are compact typed brain events derived from
`PerformanceEpisodeV3` timelines. Supported categories are `active_project`,
`user_preference`, `unresolved_commitment`, `correction`, `visual_event`,
`repeated_frustration`, and `success_pattern`. Supported behavior effects are
`shorter_explanation`, `project_constraint_recall`, `corrected_preference`,
`avoid_repetition`, `visual_context_acknowledgement`, `tentative_callback`,
`suppressed_stale_memory`, and `none`.

Default payloads do not include raw memory bodies, raw transcripts, hidden
prompts, full messages, secrets, media data, SDP, or ICE candidates.

## Cross-Language Continuity

Retrieval expands query terms through a deterministic zh/en bridge for Blink
product terms and primary local browser concepts:

- Chinese/Melo/browser/Moondream/WebRTC terms can influence English Kokoro
  replies when relevant.
- English/Kokoro/browser/Moondream/WebRTC terms can influence Chinese Melo
  replies when relevant.

Selected cross-language memories expose a bounded display summary. Known terms
use deterministic phrase mappings; otherwise the trace says that a
cross-language memory was selected without dumping hidden context.

## Runtime Surfaces

The current continuity trace and discourse episode summaries are additive on:

- `/api/runtime/memory` as `memory_continuity_trace`;
- `/api/runtime/actor-state.memory_persona.memory_continuity_trace`;
- memory/persona performance planning through `PerformancePlanV2` and
  `PerformancePlanV3`;
- `/api/runtime/operator` memory payloads as memory effects and selected
  discourse episode refs.

Actor/performance events only receive counts, display-kind buckets, category
labels, effect labels, selected discourse episode IDs, `memory_effect`,
`cross_language_count`, and reason codes.

## User Commands

The bounded memory tool prompt and tools now cover:

- remember / 记住;
- forget / 忘记;
- correct / 更正;
- what do you remember / 你现在记得什么;
- why did you answer that way / 为什么这样回答.

`brain_list_visible_memories` remains the visible memory read path.
`brain_explain_memory_continuity` returns the latest public trace summary for
"why did you answer that way" without exposing hidden prompts or raw memory
records.

## Behavior

The Phase 10 performance compiler accepts a continuity trace with V3 discourse
refs. Selected memory changes the memory callback policy, response shape, and UI
style hints. Concise preferences shorten answer shape, project constraints bias
planning, corrections trigger repair wording, repeated frustrations avoid
repeating failed patterns, and stale or conflicted memories are tentative or
suppressed instead of becoming contradictory current facts.

Blink keeps continuity, memory, and style without claiming a fabricated human
biography.

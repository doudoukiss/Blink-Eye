# Current Project Snapshot

## Research Alignment

Blink's V3 performance-intelligence direction is informed by the GenJen paper
set summarized in [../research_references.md](../research_references.md),
especially the LPM framing of conversation as real-time performance. The source
tree already reflects those ideas through separated speaking/listening control,
public actor state, bounded TTS lookahead, camera honesty, persona/memory
planning, and benchmarked bilingual browser paths.

This is why V3 is not just another chat layer. It is an attempt to make
frontier conversational-agent ideas practical, local, inspectable, and safe to
iterate.

## Equal primary local browser paths

- Chinese conversation path: `./scripts/run-local-browser-melo.sh`
  - profile: `browser-zh-melo`
  - language: Chinese / `zh`
  - TTS: `local-http-wav/MeloTTS`
  - media: browser/WebRTC microphone + camera
  - vision: Moondream/browser vision on by default, continuous perception off by default
- English conversation path: `./scripts/run-local-browser-kokoro-en.sh`
  - profile: `browser-en-kokoro`
  - language: English / `en`
  - TTS: `kokoro/English`
  - media: browser/WebRTC microphone + camera
  - vision: Moondream/browser vision on by default, continuous perception off by default


These are equal first-class product paths. Do not improve one while silently demoting the other.

## Current Baseline In The Source Tree

The source tree already contains a bilingual actor runtime with:

- `docs/bilingual_actor_runtime/` phase documentation.
- `scripts/run-local-browser-melo.sh` for `browser-zh-melo`.
- `scripts/run-local-browser-kokoro-en.sh` for `browser-en-kokoro`.
- `/api/runtime/actor-state` and `/api/runtime/actor-events` style public runtime surfaces.
- Actor event/state schemas and compatibility performance-state endpoints.
- Conversation floor controller, WebRTC audio health, active listening, camera scene state, speech director, persona performance plan, memory continuity traces, browser actor surface, release-gate scripts, and avatar-adapter contract.
- `./scripts/eval-bilingual-actor-bench.sh` as the deterministic bilingual actor gate.

Historical docs report validation through pytest, local browser workflow tests,
the bilingual actor bench, and browser performance bench. Treat those reported
results as context, not as a replacement for rerunning gates after changes.

## What V3 Is Not

- It is not a return to native PyAudio as the primary UX path.
- It is not a realistic human avatar implementation.
- It is not a raw camera/audio recording system.
- It is not a new hidden personality prompt layer.
- It is not a unilateral Chinese/Melo or English/Kokoro optimization.

## What V3 Is

A new layer above the actor runtime:

```text
Bilingual Browser Actor Runtime
  -> performance episode ledger
  -> boundary-aware actor control scheduler
  -> semantic active listener
  -> scene-social perception state
  -> performance planner v3
  -> dual TTS speech director v3
  -> persona/memory continuity v3
  -> preference learning workbench
  -> bilingual performance bench v3
```

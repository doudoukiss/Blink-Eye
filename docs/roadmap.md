# Blink Roadmap

Blink’s roadmap is to turn the current local-first framework into a dependable
voice and multimodal assistant runtime that is easy to inspect, easy to run
locally, and serious enough to extend into production systems.

The code already has the important foundations: a frame pipeline, local text and
voice entry points, browser/WebRTC runtime, optional provider integrations,
typed memory, continuity evaluation, public-safe actor state, camera honesty,
and bounded embodied actions. The next work is about making those foundations
feel coherent, reliable, and product-grade.

The actor-runtime and performance-planning direction is informed by the GenJen
paper set summarized in [`research_references.md`](./research_references.md).

That makes Blink a research-forward project, not only an integration wrapper.
The codebase is translating frontier ideas from conversational performance
research into practical local infrastructure: visible actor state, separated
speaking/listening streams, multi-reference persona anchors, interruption-safe
speech queues, camera honesty, and benchmark-driven product iteration.

## Current State

Blink is ready for local engineering use:

- Text chat runs through Ollama by default.
- Native voice runs through local STT and local TTS.
- Browser/WebRTC voice exposes the main daily-use product surface at
  `http://127.0.0.1:7860/client/`.
- `browser-zh-melo` and `browser-en-kokoro` are equal primary browser paths.
- The Chinese path uses MeloTTS behind the stable `local-http-wav` seam.
- The English path uses Kokoro with browser media handling.
- Camera/Moondream grounding is available on demand, with explicit public state
  for freshness and availability.
- Actor state and actor events make listening, thinking, speaking, looking,
  interruption, memory/persona use, and degradation inspectable.
- Core brain behavior has deterministic proof lanes and property/stateful test
  coverage.

The public repository is intentionally source-first. Generated browser package
copies, runtime logs, local builds, virtual environments, and internal planning
bundles are excluded; required runtime assets that Blink loads directly are
retained.

## Product Ambition

Blink should feel like a capable local companion for technical work: responsive,
interruptible, honest about what it sees, able to remember useful context, and
legible enough that an operator can understand why it behaved the way it did.

The ambition is world-class in architecture even while remaining local-first:
Blink should make advanced assistant behavior inspectable, reproducible, and
safe enough to improve through evidence rather than hidden prompt churn.

That ambition breaks down into five commitments:

- **Local trust first:** a user should be able to run and inspect meaningful
  behavior without opening accounts with hosted providers.
- **Bilingual quality:** Chinese and English browser paths should both receive
  first-class treatment, with Mandarin speech quality handled as a real product
  problem rather than a prompt trick.
- **Observable autonomy:** runtime state, memory use, camera state, and
  performance decisions should be visible through public-safe APIs.
- **Bounded embodiment:** robot/head actions and future avatar surfaces should
  remain explicit, narrow, replayable, and easy to disable.
- **Evidence-driven improvement:** performance learning should use structured
  local evidence and release gates before changing behavior.

## Research Translation Priorities

The most important paper ideas Blink carries forward are:

- **Performance trilemma awareness:** real assistants need expressiveness,
  latency control, and long-horizon consistency at the same time.
- **Full-duplex conversation:** listening is active behavior, not silence while
  waiting for a transcript.
- **Online actor state:** a browser assistant should expose whether it is
  listening, thinking, looking, speaking, repairing, or degraded.
- **Multi-reference identity translated to persona:** stable behavior comes
  from selected situational anchors, not one monolithic prompt.
- **Boundary-aligned updates:** interruption, camera, memory, and speech changes
  should apply at safe runtime boundaries instead of corrupting in-flight output.
- **Performance evaluation:** release gates should score state clarity,
  felt-heard, interruption naturalness, speech pacing, camera honesty, memory
  usefulness, persona consistency, and not-fake-human behavior.

## Near-Term Priorities

1. **Polish the browser experience**

   Keep `/client/` fast, readable, and useful for repeated daily interaction.
   The actor/workbench surface should make the runtime easier to trust without
   overwhelming the conversation.

2. **Strengthen Chinese voice quality**

   Continue improving MeloTTS sidecar ergonomics, chunking, normalization,
   fallback behavior, and comparison tooling while keeping the main package
   dependency graph clean.

3. **Make memory behavior obviously useful**

   Keep memory preview-first, governed, and public-safe. Memory should change
   behavior in understandable ways without exposing private raw records in
   traces or UI payloads.

4. **Harden interruption and turn-taking**

   Preserve protected playback defaults, bounded TTS queues, and explicit
   repair states. Regressions where the first answer works and the second turn
   fails should be treated as release blockers.

5. **Keep proof lanes fast**

   The canonical brain and embodied proof scripts should remain small enough to
   run during normal development, with broader provider/browser/audio checks
   reserved for changes that need them.

## Longer-Term Direction

- Adaptive performance learning from local episode evidence.
- Stronger scene and situation modeling without raw media retention.
- Richer operator workbench controls for memory, teaching, policy proposals,
  and safe rollout.
- Avatar-adapter contracts that consume public actor state without realistic
  human likeness, identity cloning, or raw camera/audio export.
- Optional hosted-provider lanes that preserve the local-first architecture
  instead of becoming required infrastructure.

## Non-Goals

- Making hosted providers mandatory for local development.
- Committing generated browser bundles, runtime logs, model caches, or virtual
  environments.
- Replacing the frame pipeline with a single chatbot loop.
- Pretending camera or memory capabilities exist when the runtime has no fresh
  evidence.
- Adding broad autonomous physical control without bounded policies and replay.

## Where To Read Next

- [`../README.md`](../README.md)
- [`../LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md)
- [`USER_MANUAL.md`](./USER_MANUAL.md)
- [`chinese-conversation-adaptation.md`](./chinese-conversation-adaptation.md)
- [`bilingual_actor_runtime/README.md`](./bilingual_actor_runtime/README.md)
- [`FILE_MAP.md`](./FILE_MAP.md)

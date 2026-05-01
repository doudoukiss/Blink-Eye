# Research References

This page records the research papers and paper themes from the GenJen reference
set that informed Blink's current architecture. These papers are design
inspiration, not vendored implementation. Blink does not copy paper code,
datasets, model weights, prompts, or private artifacts from the reference set.

The point of this page is also positioning: Blink is designed as a modern,
research-forward assistant runtime. It takes ideas that are appearing in
frontier conversational character-performance papers and turns them into
inspectable local software instead of leaving them as hidden prompts or demo
videos.

## Primary Paper Used Directly

### LPM 1.0: Video-Based Character Performance Model

- Source: [arXiv:2604.07823](https://arxiv.org/abs/2604.07823)
- Paper page: [Hugging Face Papers](https://huggingface.co/papers/2604.07823)

The LPM paper frames conversation as a full-duplex performance problem:
characters speak, listen, react, emote, and preserve identity over time. Its
central ideas for Blink are the performance trilemma, separate speaking and
listening control streams, identity-aware multi-reference conditioning,
boundary-aligned online updates, and evaluation through an interaction benchmark
rather than answer correctness alone.

These ideas matter because they describe the gap between a normal chatbot and a
high-quality interactive agent. A chatbot can wait for a message and answer. A
performance-oriented agent needs to show that it is listening, keep speech
interruptible, preserve a coherent identity over many turns, use memory without
leaking private state, and evaluate whether the interaction actually felt good.

Blink translates those ideas into a local assistant runtime:

- LPM speaking/listening branches become Blink's separated speech-output and
  user-listening control paths.
- LPM online runtime states become Blink actor state and actor events.
- LPM multi-reference identity becomes a persona reference bank and memory-aware
  performance planning.
- LPM boundary-aligned generation becomes bounded speech chunks, stale-token
  invalidation, and interruption-safe TTS queues.
- LPM-Bench becomes the Blink bilingual actor bench, with metrics for
  state clarity, felt-heard, interruption naturalness, camera honesty, memory
  usefulness, persona consistency, and not-fake-human behavior.

The scope boundary is important: Blink does not attempt to train or ship a
realistic human video avatar. The current product uses public-safe actor state,
abstract browser UI, local voice paths, and optional bounded embodiment.

This boundary is part of the engineering quality. Blink adopts the interaction
architecture behind the paper while avoiding unsafe or premature claims about
human likeness, identity cloning, or proprietary video generation.

## Adjacent Themes From The GenJen Paper Set

The broader GenJen research set influenced priorities rather than direct
implementation details:

- **Streaming audio-video generation:** preserve low-latency, boundary-aware
  control instead of building long uninterruptible response queues.
- **Talking portrait and avatar systems:** define avatar-ready contracts, but
  keep realistic human likeness out of the default product.
- **Identity and persona consistency:** use multiple situational references
  instead of one hidden personality prompt.
- **Long-term conversational memory:** make memory observable, correctable, and
  behaviorally meaningful rather than a private transcript pile.
- **Interruption and dialogue repair:** treat overlap, barge-in, repair, and
  resumption as first-class runtime states.
- **Evaluation and preference learning:** use structured traces, scorecards,
  and preference records before changing product behavior.

## Where The Ideas Appear In Blink

- [01_research_translation_lpm_to_blink.md](./01_research_translation_lpm_to_blink.md)
- [bilingual_actor_runtime/01_lpm_research_translation.md](./bilingual_actor_runtime/01_lpm_research_translation.md)
- [performance_intelligence_v3/01_lpm_research_translation.md](./performance_intelligence_v3/01_lpm_research_translation.md)
- [bilingual_actor_runtime/README.md](./bilingual_actor_runtime/README.md)
- [roadmap.md](./roadmap.md)

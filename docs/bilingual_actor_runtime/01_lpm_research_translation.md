# LPM research translation for Blink

Source context: this note translates the LPM paper from the GenJen reference
set. See [../research_references.md](../research_references.md) for the
maintained paper summary and scope boundaries.

LPM 1.0 argues that conversation is a performance problem, not just a talking problem. Its most transferable ideas are:

1. **Speaking and listening are distinct control streams.** Blink should separately model assistant speech output and user-listening input.
2. **Runtime state is part of the model.** Warmup, idle, listening, responding, and boundary-aligned updates matter as much as text generation.
3. **Identity stability requires multiple references.** Blink should use multi-reference persona examples, not only a single personality prompt.
4. **Listening behavior is scarce and important.** Blink must visibly listen during user speech instead of appearing dead until it answers.
5. **Evaluation must measure performance.** We should test felt-heard, interruption, camera grounding, memory visibility, personality consistency, and enjoyment, not only answer correctness.

## Blink-specific translation

```text
LPM visual performance engine
→ Blink browser actor runtime

speaking audio branch
→ Melo/Kokoro speech performance director

listening audio branch
→ Active Listener + Conversation Floor Controller

multi-reference identity images
→ persona reference bank + memory/persona performance compiler

online interactive video runtime
→ browser/WebRTC actor event loop and state API

LPM-Bench
→ Blink Bilingual Actor Bench
```

## Important scope boundary

This upgrade does **not** build a realistic human avatar. It creates the actor-event and performance-plan substrate that a future avatar can consume safely.

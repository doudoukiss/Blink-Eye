# Next huge upgrade strategy

## Upgrade name

Blink Bilingual Performance Intelligence V3.

## Strategic objective

Move from an observable actor runtime to an adaptive performance intelligence layer. The system should not merely report state; it should use state to shape behavior in real time.

## Product outcomes

A successful V3 build should make Blink feel better in these concrete ways:

- During long user speech, Blink visibly tracks topic, constraints, uncertainty, and readiness.
- When the user interrupts, Blink reacts according to floor state and echo safety rather than crude VAD alone.
- When Blink speaks, output is chunked, subtitled, paced, and interruptible.
- When the camera is used, Blink shows whether it is looking, what frame freshness policy applies, and whether vision was unavailable or stale.
- Memory and persona are visible through behavior: response length, callback choice, repair stance, disagreement style, and project continuity.
- Chinese/Melo and English/Kokoro remain equal and comparable.
- Dogfooding produces preference data that guides future policy changes.

## Engineering principles

1. Preserve the current actor runtime contracts.
2. Add V3 capabilities as adapters and extensions, not rewrites.
3. Keep all persistent traces public-safe by default.
4. Treat low-level turn-taking as deterministic policy, not LLM free choice.
5. Keep continuous perception off by default.
6. Keep protected playback default unless echo-safe or explicitly armed.
7. Do not claim unsupported TTS controls.
8. Do not implement realistic human likeness or identity cloning in this tranche.

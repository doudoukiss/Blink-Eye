# 05 — Blink-Actor-Bench evaluation plan

Blink-Actor-Bench adapts LPM-Bench’s performance-evaluation spirit to a browser voice/camera agent. The goal is not to score visual photorealism. The goal is to score whether Blink behaves like a coherent, responsive, enjoyable conversational participant.

## Evaluation layers

### Automated trace validation

Validate actor traces for:

- missing speech-start / speech-stop / final-transcript events;
- stale mode after TTS end;
- accepted interruption without audio invalidation;
- camera answer without frame seq/age when vision was used;
- memory-use claim without memory trace;
- unsafe barge-in enabled under echo-risk state;
- raw audio/image/SDP/token leakage in default trace.

### Scripted interaction tests

Test suites:

```text
listening_long_turn
speaking_pacing
interruption_real
interruption_backchannel
camera_grounding
memory_continuity
persona_consistency
repair_and_correction
long_session_stability
```

### Human scoring

Use 1–5 Likert ratings:

```text
felt_heard
state_clarity
interruption_naturalness
speech_pacing
camera_grounding
memory_usefulness
persona_consistency
enjoyment
not_fake_human
not_annoying
```

### Pairwise G/S/B

Compare current build vs previous build:

```text
Good: current is better
Same: no meaningful difference
Bad: previous is better
```

Pairwise comparison should be used for demos and major upgrades because it is easier to judge “better/worse” than absolute quality.

## Release gate proposal

Initial local thresholds:

```text
state_clarity >= 4.0
felt_heard >= 3.8
interruption_naturalness >= 3.5 when barge-in is enabled
speech_pacing >= 3.8
camera_grounding >= 3.8 for vision tasks
memory_usefulness >= 3.5 when seeded memory is relevant
enjoyment >= 3.5
no critical privacy leaks
no unsafe camera/vision claims
```

These thresholds should be tightened after dogfooding.

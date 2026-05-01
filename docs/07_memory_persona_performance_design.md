# 07 — Memory and Persona Performance Design

## Problem

Blink may have memory and personality internally, but users do not feel them unless they change observable behavior.

## Design

Each assistant turn should produce a small `PerformancePlan` and optional `MemoryUseTrace`.

The plan should decide:

```text
stance
response shape
memory callback
camera policy
interruption policy
subtitle/speech chunking
follow-up policy
```

The trace should expose:

```text
which memories were selected
which memories were suppressed
why the memory mattered
what behavior changed
```

## Browser UI

Show only short summaries:

```text
Used in this reply:
- Current project path: browser/WebRTC + MeloTTS + camera.
- User preference: avoid overbuilding native PyAudio barge-in.
- Style: direct implementation roadmap.
```

## Persona references

Do not rely on one giant personality prompt. Maintain reference examples for:

```text
interruption
camera use
memory callback
correction
technical deep dive
concise answer
disagreement
uncertainty
playfulness without fake humanity
```

## Constraint

Blink should feel like a coherent AI presence, not a fake human with fabricated biography.

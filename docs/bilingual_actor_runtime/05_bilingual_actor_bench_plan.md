# Blink Bilingual Actor Bench plan

## Profiles

```text
browser-zh-melo      Chinese conversation, MeloTTS, Moondream/browser vision
browser-en-kokoro    English conversation, Kokoro, Moondream/browser vision
```

## Functional categories

- Connection and profile readiness
- Active listening
- Conversation floor and turn-taking
- Adaptive interruption and echo health
- Speech performance and subtitles
- Camera/Moondream grounding
- Memory/persona behavior
- Recovery/degradation
- Long-session continuity
- Cross-profile parity

## Human rating dimensions

Use 1–5 Likert scores and optional pairwise preference:

```text
state clarity
felt-heard
voice pacing
camera grounding
memory usefulness
interruption naturalness
personality consistency
enjoyment
not fake-human
not annoying
```

## Hard release blockers

- Unsafe trace payload
- Camera claim without camera grounding
- Hidden camera use
- Self-interruption in protected playback
- Stale TTS after accepted interruption
- Memory contradiction after correction
- Profile parity regression
- English path accidentally disables Moondream by default
- Chinese path accidentally bypasses MeloTTS sidecar health checks

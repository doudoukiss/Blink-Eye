# Evaluation and release gate

## Bench name

`BilingualPerformanceBenchV3`

## Matched profile matrix

Every deterministic test category must run against both profiles:

```text
browser-zh-melo  -> Chinese + MeloTTS + Moondream
browser-en-kokoro -> English + Kokoro + Moondream
```

## Evaluation categories

- connection and state visibility
- active listening during long user turns
- speech performance and subtitles
- overlap, interruption, and repair
- camera/Moondream grounding and no-vision fallback
- memory callback and correction
- persona consistency across languages
- long-session continuity
- preference comparison and dogfooding data
- privacy and avatar-adapter safety

## Human-facing scores

Use 1-5 ratings and pairwise comparison where possible:

- state clarity
- felt heard
- voice pacing
- interruption naturalness
- camera honesty
- memory usefulness
- persona consistency
- enjoyment
- not fake-human

## Hard blockers

Any of these fails the gate:

- one primary path regresses or loses equal status
- hidden camera use
- false camera claim
- raw audio/image stored in default traces
- SDP/ICE/secret/prompt leakage
- self-interruption in protected playback
- stale TTS chunk after accepted interruption
- unsupported TTS capability claim
- memory contradiction after correction
- missing consent or opt-out controls
- realistic human avatar or identity-cloning capability in this tranche

## Commands to preserve

This plan expects future implementation to keep and extend these commands:

```bash
./scripts/eval-bilingual-actor-bench.sh
uv run --extra runner --extra webrtc pytest tests/test_local_workflows.py
uv run pytest
```

Add V3-specific scripts only after Phase 01 and Phase 02 define stable fixtures and schemas.

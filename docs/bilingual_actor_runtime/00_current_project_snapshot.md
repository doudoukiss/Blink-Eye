# Current Project Snapshot

## Equal primary browser paths

Blink treats the following as equal first-class product paths:

```bash
./scripts/run-local-browser-melo.sh       # Chinese conversation, MeloTTS, Moondream/browser vision
./scripts/run-local-browser-kokoro-en.sh  # English conversation, Kokoro, Moondream/browser vision after Phase 01
```

Both paths use:

```text
Browser/WebRTC microphone and camera
MLX Whisper or the configured local STT backend
Local/hybrid LLM configuration
Browser audio playback
Moondream/browser vision grounding
http://127.0.0.1:7860/client/
```

## Maintained Repository State

- `scripts/run-local-browser-melo.sh` sets the `browser-zh-melo` profile,
  enables browser vision by default, starts or reuses the MeloTTS HTTP-WAV
  sidecar, and labels the TTS runtime as `local-http-wav/MeloTTS`.
- `scripts/run-local-browser-kokoro-en.sh` sets the `browser-en-kokoro`
  profile, starts in English, uses Kokoro, keeps browser vision on by default,
  leaves continuous perception off, and keeps protected playback enabled.
- `src/blink/cli/local_browser.py` owns the browser runtime, public APIs,
  actor state, memory/operator surfaces, and `/client/` mounting.
- `src/blink/interaction/` contains event, state, active-listening, floor,
  WebRTC health, and camera-honesty primitives that should be extended rather
  than duplicated.
- `src/blink/brain/evals/browser_perf_bench.py` and the bilingual actor bench
  keep parity pressure on both primary browser profiles.

## Non-primary paths

Native PyAudio paths remain useful for backend isolation, especially for STT, LLM, and TTS debugging. They should not drive daily UX decisions because native laptop-speaker barge-in can self-interrupt without browser/WebRTC echo cancellation.

## Operating Principle

Do not improve one primary path while silently demoting the other. Chinese
Melo/Moondream and English Kokoro/Moondream are parallel targets throughout the
browser runtime.

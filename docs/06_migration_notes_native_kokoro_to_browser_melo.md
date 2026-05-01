# 06 — Migration Notes from Native Kokoro Plan to Primary Browser Plans

`browser-zh-melo` and `browser-en-kokoro` are now equal primary browser/WebRTC
product paths. The older migration from native Kokoro camera UX to browser Melo
is still useful history, but it no longer means Melo is the only primary browser
route.

## Removed as primary scope

- Native Kokoro camera path as the main product path.
- Native terminal/TUI interaction layer as the main feedback surface.
- Native PyAudio barge-in as a default interaction mode.

## Retained concepts

- Performance event bus.
- Explicit listening/thinking/speaking/looking/interruption states.
- Active listening.
- Memory/persona visibility.
- Speech director and cancellation safety.
- Evaluation harness inspired by LPM-Bench.

## Changed technical targets

| Old target | New target |
|---|---|
| `scripts/run-local-voice-kokoro-camera-en.sh` | `scripts/run-local-browser-melo.sh` for Chinese camera UX; `scripts/run-local-browser-kokoro-en.sh` for English browser Kokoro UX |
| native PyAudio | browser/WebRTC for daily product paths |
| Kokoro-only product UX | MeloTTS via local-http-wav for Chinese, Kokoro for English browser voice |
| local control panel separate from media | `/client/` is media + feedback shell |
| native camera helper as product UX | browser camera + `fetch_user_image` for `browser-zh-melo`; native helper remains isolation-only |
| native interaction state | browser interaction state for product paths; native status lines for backend isolation |

## Regression rule

Any future change that enables barge-in by default must include echo-safety validation and tests. The previous Mac speaker self-interruption failure must be treated as a known regression pattern.

Use [`debugging/native_voice_isolation.md`](./debugging/native_voice_isolation.md)
for the current native source of truth.

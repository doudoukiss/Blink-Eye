# Chinese TTS Upgrade

This document explains the Chinese TTS side of the local Blink adaptation and
the concrete upgrade path now included in this repo.

For the full system-level record, including speech shaping, interruption
handling, docs realignment, and native playback reliability, see
[`docs/chinese-conversation-adaptation.md`](./docs/chinese-conversation-adaptation.md).

## Diagnosis

The current local Chinese bootstrap voice path is:

- STT: `mlx-whisper`
- LLM: Ollama
- TTS: `kokoro`
- default Mandarin voice: `zf_xiaobei`

That path is useful because it runs directly on an Apple Silicon MacBook Pro with no extra voice infrastructure. It is **not** the Chinese-quality target anymore.

The diagnosis captured in [`docs/chinese-conversation-adaptation.md`](./docs/chinese-conversation-adaptation.md)
and [`docs/roadmap.md`](./docs/roadmap.md) is:

- the main bottleneck is Mandarin TTS quality plus a weak speech frontend
- the core frame pipeline is not the main problem
- Chinese voice quality should improve through the existing `local-http-wav` seam, not through a core architecture rewrite

That diagnosis still holds, but one runtime issue needed a separate fix outside
the sidecar itself:

- the native local voice path could cut itself off because speaker bleed caused
  self-interruption
- that fix lives in the local CLI path, not in the Melo server

## Why MeloTTS Is The Primary Upgrade

For this repo, the first Chinese-quality path to try is now **MeloTTS via `local-http-wav`**.

Why this fits:

- Chinese and mixed Chinese-English speech are core MeloTTS use cases
- the upstream project presents CPU real-time inference as a supported mode
- its installation story is lighter than CosyVoice
- it fits the existing Blink `POST /tts` contract cleanly

Why it is not a direct Blink backend:

- MeloTTS pins older transformer dependencies upstream
- this repo already uses a newer transformer range in the main package
- an in-process backend would create dependency conflict and maintenance drag

So the implementation direction is:

- keep `BLINK_LOCAL_TTS_BACKEND` unchanged at the product surface
- keep Blink talking to `local-http-wav`
- auto-prefer a healthy `local-http-wav` sidecar for Chinese voice/browser
  runtime when no backend is explicitly pinned
- run MeloTTS in its own isolated Python `3.11` environment
- keep the Melo server outside `src/blink`

## What Changed

This repo now includes a repo-owned MeloTTS HTTP-WAV server and helper scripts:

- server module: [`local_tts_servers/melotts_http_server.py`](./local_tts_servers/melotts_http_server.py)
- bootstrap helper: [`local_tts_servers/melotts_reference.py`](./local_tts_servers/melotts_reference.py)
- bootstrap script: [`scripts/bootstrap-melotts-reference.sh`](./scripts/bootstrap-melotts-reference.sh)
- server runner: [`scripts/run-melotts-reference-server.sh`](./scripts/run-melotts-reference-server.sh)
- native voice wrapper: [`scripts/run-local-voice-melo.sh`](./scripts/run-local-voice-melo.sh)
- browser wrapper: [`scripts/run-local-browser-melo.sh`](./scripts/run-local-browser-melo.sh)
- tested runtime pins: [`docs/MeloTTS-reference/runtime-requirements.txt`](./docs/MeloTTS-reference/runtime-requirements.txt)

The server:

- exposes `GET /healthz`
- exposes `GET /voices`
- exposes `POST /tts`
- returns mono `24 kHz` WAV bytes for Blink
- keeps language and speaker selection separate
- caches MeloTTS models per language and device for responsiveness

In parallel, the repo-local Chinese conversation path was hardened outside the
Melo server itself:

- speech-safe prompts were separated from text-chat prompts
- Chinese technical text normalization was expanded before TTS
- native local voice now defaults to protected playback with opt-in barge-in
- `local-http-wav` no longer reuses Kokoro or Piper voice env vars implicitly
- when a configured `local-http-wav` speaker is stale and the sidecar exposes
  `/voices`, Blink now warns and falls back to the sidecar default speaker
  instead of failing the first spoken reply
- Chinese voice and browser runtime now auto-switch from Kokoro to
  `local-http-wav` when the sidecar is healthy and the backend was not pinned

This keeps the Blink runtime unchanged:

- runtime side: existing `local-http-wav` backend stays unchanged
- quality upgrade side: MeloTTS stays isolated in its own environment
- repo bootstrap side: the repo patches upstream MeloTTS for the zh/en Apple Silicon use case instead of asking users to hand-fix stale packaging, dead model URLs, and broken CPU routing

## Start The Recommended Upgrade Path

### 1. Bootstrap the isolated MeloTTS environment

```bash
./scripts/bootstrap-melotts-reference.sh
```

That creates an isolated virtualenv under `docs/MeloTTS-reference/.venv`.
It also creates a patched upstream checkout under `docs/MeloTTS-reference/vendor/MeloTTS` and prefetches the zh/en assets used by the local sidecar.

### 2. Start the MeloTTS HTTP-WAV server

```bash
./scripts/run-melotts-reference-server.sh
```

By default it serves:

```text
http://127.0.0.1:8001
```

Useful overrides:

```bash
BLINK_LOCAL_MELO_HOST=127.0.0.1 \
BLINK_LOCAL_MELO_PORT=8001 \
BLINK_LOCAL_MELO_DEVICE=cpu \
BLINK_LOCAL_MELO_SPEAKER_ZH=ZH \
BLINK_LOCAL_MELO_SPEAKER_EN=EN-US \
BLINK_LOCAL_MELO_SPEED=1.0 \
./scripts/run-melotts-reference-server.sh

BLINK_LOCAL_MELO_PREFETCH=0 ./scripts/bootstrap-melotts-reference.sh
```

### 3. Point Blink at the server

Use the direct local-http-wav path:

```bash
BLINK_LOCAL_TTS_BACKEND=local-http-wav \
BLINK_LOCAL_HTTP_WAV_TTS_BASE_URL=http://127.0.0.1:8001 \
./scripts/run-blink-voice.sh
```

Or use the repo wrappers:

```bash
./scripts/run-local-voice-melo.sh
./scripts/run-local-browser-melo.sh
```

Or just run the normal voice/browser entrypoints without pinning a backend:
when Chinese is selected and the sidecar is healthy, Blink now prefers it
automatically and keeps Kokoro as the fallback.

## Evaluate Quality

Keep one reproducible Kokoro baseline:

```bash
./scripts/eval-local-tts.sh --backend kokoro --language zh --all-kokoro-zh-voices
```

Then compare the MeloTTS sidecar path:

```bash
./scripts/eval-local-tts.sh --backend local-http-wav --language zh --voice-zh ZH
```

Outputs are written under:

```text
artifacts/tts-eval/
```

## Advanced Secondary Path: CosyVoice

CosyVoice remains available as an advanced optional sidecar when you want to compare another stronger Mandarin backend:

- adapter: [`src/blink/cli/local_cosyvoice_adapter.py`](./src/blink/cli/local_cosyvoice_adapter.py)
- wrappers:
  - [`scripts/run-local-cosyvoice-adapter.sh`](./scripts/run-local-cosyvoice-adapter.sh)
  - [`scripts/run-local-browser-cosyvoice.sh`](./scripts/run-local-browser-cosyvoice.sh)
  - [`scripts/run-local-voice-cosyvoice.sh`](./scripts/run-local-voice-cosyvoice.sh)

CosyVoice reference source is optional external material and is not committed in
the source-first repository.

## Stop

Stop the MeloTTS server with `Ctrl+C` in its terminal.

If you want to stop it by process name:

```bash
pkill -f 'melotts_http_server|run-melotts-reference-server' || true
```

## Notes

- This path is intentionally Chinese-first.
- The bootstrap in-repo path is still `kokoro`, now with a speech-safe Chinese prompt and stronger deterministic normalization.
- The product runtime path for Chinese voice/browser will prefer Melo
  automatically when the sidecar is healthy, unless you explicitly pin Kokoro.
- If you want English through the Melo sidecar, set `BLINK_LOCAL_MELO_SPEAKER_EN` explicitly if the default does not fit your machine.
- The repo-owned bootstrap keeps Melo on CPU by default on Apple Silicon because the upstream MPS path is not the stable local voice path.
- If MeloTTS is not enough for your quality target, keep the same `local-http-wav` seam and evaluate XTTS, CosyVoice, or another sidecar later.
- CosyVoice reference checkouts remain external design material, not packaged runtime dependencies.

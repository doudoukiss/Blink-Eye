# Phase 01 dual-path baseline

Phase 01 makes the two local browser product paths equal first-class lanes:

```bash
./scripts/run-local-browser-melo.sh       # browser-zh-melo, Chinese, MeloTTS
./scripts/run-local-browser-kokoro-en.sh  # browser-en-kokoro, English, Kokoro
```

Both launchers use the same browser client:

```text
http://127.0.0.1:7860/client/
```

## Baseline contract

| Path | Profile | Language | TTS label | Browser vision | Continuous perception | Playback |
|---|---|---:|---|---:|---:|---|
| Chinese | `browser-zh-melo` | `zh` | `local-http-wav/MeloTTS` | on | off | protected |
| English | `browser-en-kokoro` | `en` | `kokoro/English` | on | off | protected |

The English Kokoro browser path now keeps Moondream/browser vision available by
default. Disable it only with `--no-vision` or `BLINK_LOCAL_BROWSER_VISION=0`.

Both wrappers print a startup banner with profile, language, TTS runtime label,
WebRTC, camera/vision status, continuous perception, protected playback,
barge-in policy, log file, and client URL. The lower-level browser runtime also
prints a readiness summary after `/client/` is actually serving.

The V3 Phase 01 baseline snapshot is checked in at:

```text
evals/bilingual_performance_intelligence_v3/baseline_profiles.json
```

It is regenerated deterministically from checked-in runtime profile defaults by:

```bash
./scripts/eval-performance-intelligence-baseline.sh
```

Use `./scripts/eval-performance-intelligence-baseline.sh --check` in release
verification to ensure profile, language, TTS label, browser vision default,
continuous perception default, protected playback default, actor runtime
endpoints, and gate commands have not drifted.

## Compatibility boundary

Native PyAudio paths are backend isolation lanes, not product UX lanes. Use
them to isolate microphone, STT, LLM, TTS, or the macOS camera helper. Do not
use native PyAudio behavior to decide the browser product defaults because it
lacks the browser/WebRTC media permission and echo-cancellation model.

The primary browser paths keep protected playback on unless the user explicitly
arms barge-in through `--allow-barge-in` or `BLINK_LOCAL_ALLOW_BARGE_IN=1`.

## Safety

Phase 01 does not add persistent raw media traces. Runtime state and startup
banners stay bounded to profile, language, backend labels, public media state,
log path, and client URL. Do not store raw audio, raw images, SDP, ICE
candidates, secrets, hidden prompts, full messages, or unbounded transcripts in
default traces.

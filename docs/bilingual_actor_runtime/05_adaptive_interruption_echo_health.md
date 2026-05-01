# Adaptive Interruption And WebRTC Echo Health

Phase 05 adds a public-safe WebRTC audio-health snapshot and conservative
interruption policy for both primary browser paths:

- `browser-zh-melo`: Chinese, MeloTTS through `local-http-wav`, Moondream
- `browser-en-kokoro`: English, Kokoro, Moondream

Protected playback remains the default. Echo cancellation and noise suppression
lower the risk label, but they do not by themselves prove that speaker playback
is safe for live barge-in.

## Public State

`GET /api/runtime/actor-state` now includes `webrtc_audio_health`.

Important fields:

- `microphone_state`: public microphone readiness such as `ready`, `receiving`,
  `stalled`, or `permission_denied`
- `input_track_state`: bounded WebRTC input-track health
- `output_playback_state`: `idle`, `playing`, `speaking`, `stalled`, `error`,
  or `unknown`
- `echo`: public browser echo hints, including echo cancellation, noise
  suppression, auto-gain control, and optional `echo_safe`
- `stats.summary`: bounded numeric WebRTC audio stats only
- `echo_risk`: `unknown`, `low`, `medium`, or `high`
- `barge_in_state`: `protected`, `armed`, or `adaptive`

Unsafe payloads are not allowed in state or traces: raw audio, raw images, SDP,
ICE candidates, device labels, credentials, prompts, model messages, and full
transcripts are omitted.

## Policy

Speaker mode starts as `protected`. While protected, user speech during assistant
speech is surfaced as an interruption candidate or suppression reason instead
of cutting off the assistant.

Live interruption becomes eligible only when:

- the user explicitly starts the runtime with `--allow-barge-in`, or
- the browser/client reports a clear public echo-safe signal and the microphone
  and input track are healthy.

Short backchannels such as `嗯`, `对`, `ok`, `yeah`, and `right` remain continue
signals. Cough/noise, very brief VAD bursts, low-confidence transcripts, and
protected-playback speech are counted as false-interruption categories.

Accepted interruptions emit:

- v1 compatibility: `interruption.accepted`
- v2 actor event: `interruption_accepted`
- stale output handling: `interruption.output_flushed`
- v1 compatibility for older tools: `interruption.output_dropped`

## Bilingual Parity

The policy is shared by both primary browser profiles. Language-specific phrase
classification comes from the conversation floor controller, while echo risk and
barge-in state use the same deterministic WebRTC audio-health controller.

The two profiles should differ only in labels such as profile, language, and TTS:

- `browser-zh-melo`, `zh`, `local-http-wav/MeloTTS`
- `browser-en-kokoro`, `en`, `kokoro/English`

## Regression Fixture

The interruption/echo regression cases live in:

```bash
evals/bilingual_actor_bench/regression_interruption_echo.jsonl
```

They cover protected speaker mode, explicit echo-safe/headphone interruption,
Chinese and English backchannels, cough/noise rejection, and stale-output
flushing after accepted interruption.

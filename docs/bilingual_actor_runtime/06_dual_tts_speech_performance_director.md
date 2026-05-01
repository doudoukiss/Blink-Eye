# Phase 06/08: Dual TTS Speech Performance Director

Phase 06 makes the browser speech path chunk-aware for both primary local
products:

- `browser-zh-melo`: Browser/WebRTC + `local-http-wav/MeloTTS`
- `browser-en-kokoro`: Browser/WebRTC + `kokoro/English`

Both paths emit the same live speech chunk contract before text is sent toward
TTS. The browser client can show subtitles immediately, and interruption guards
can drop stale chunks after an accepted interruption.

## Chunk Contract

`SpeechPerformanceChunk` is the shared frame-local payload. It includes:

- `speech_director_version: 3` beside the compatibility `schema_version: 1`
- stable IDs: `chunk_id`, `generation_token`, `turn_id`, `chunk_index`
- routing labels: `language`, `tts_backend`, `role`
- live-only text: `text`, `display_text`
- playback hints: `pause_after_ms`, `interruptible`
- deterministic metadata: `estimated_duration_ms`, `subtitle_timing`,
  `stale_generation_token`, and `backend_capabilities`
- bounded diagnostics: `reason_codes`

`BrainSpeechChunk` remains as a compatibility alias. Existing callers that use
`speak_text` and `generation_id` continue to work, but new code should use
`text` and `generation_token`.

## Backend Modes

Melo uses `melo_chunked`. The director keeps the existing moderate bounds so
local-http-wav requests stay stable and Chinese punctuation remains a preferred
boundary.

Kokoro uses `kokoro_chunked` by default in `browser-en-kokoro`. Short replies
stay intact at response end. Longer English answers split on sentence
boundaries around a 120 character target with a 180 character hard maximum.
`kokoro_passthrough` is still accepted as a legacy input mode, but resolves to
`kokoro_chunked`.

Phase 08 centralizes runtime budgets as `SpeechChunkBudgetV3`. Backend defaults
remain unchanged: Melo targets 160 characters with an 80 character minimum and
220 character hard maximum; Kokoro targets 120 characters with a 40 character
minimum and 180 character hard maximum. Per-turn `PerformancePlanV3` budget
hints may shorten those numbers, but they are clamped to the backend hard
maximums and do not control low-level boundary timing.

## Privacy Boundary

Chunk `text` and `display_text` are live UI/frame data. Default performance
events and actor traces only store IDs, backend/language labels, character
counts, queue depth, duration estimates, subtitle timing labels, conservative
capability flags, pause hints, and reason codes. They do not store raw audio,
raw images, SDP, ICE candidates, secrets, hidden prompts, full messages, or
unbounded transcript/speech text.

## Interruption

On accepted interruption, buffered speech text is cleared instead of being
flushed into new TTS chunks. Already emitted stale chunks are dropped by
`BrowserInterruptedOutputGuardProcessor`, which records `output_flushed` and
the legacy `interruption.output_dropped` event for compatibility.

## Capabilities

The local capability registry has explicit conservative entries for
`local-http-wav` and `kokoro`. Both support chunk boundaries and interruption
flush. Neither claims emotional prosody, speech-rate control, pause-timing
control, partial stream abort, or hardware expression control. Phase 08 repeats
those backend capability labels on each public-safe speech chunk so offline
episode and control-frame replay can verify the runtime did not invent controls
the backend does not expose.

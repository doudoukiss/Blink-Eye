# Rollback Local Audio To Last Known Working State

This note records the recent local voice/browser changes and the safest ways to
roll back if the current runtime regresses.

## What changed most recently

### 1. Chinese STT default changed

These files were changed so Chinese voice/browser sessions no longer default to
`mlx-community/whisper-medium-mlx`:

- `src/blink/cli/local_common.py`
- `src/blink/cli/local_voice.py`
- `src/blink/cli/local_browser.py`
- `src/blink/cli/local_prefetch.py`
- `src/blink/cli/local_doctor.py`

Temporary experimental behavior that regressed the live browser path:

- Chinese + `mlx-whisper` was switched to `mlx-community/whisper-large-v3-turbo-q4`
- English + `mlx-whisper` stayed on `mlx-community/whisper-medium-mlx`
- browser and native voice startup now print the active STT backend/model

This change affects recognition quality. It should not, by itself, remove audio
playback.

What the later investigation found:

- the larger Chinese MLX Whisper model can look better on offline WAV evaluation
  but is too slow for short live browser turns on this machine
- in the browser/WebRTC path, that shows up as VAD activity without timely
  usable transcripts
- the practical symptom is that the UI keeps surfacing user-speaking events
  while the assistant never gets a stable enough transcript to answer
- the canonical default has now been restored to
  `mlx-community/whisper-medium-mlx`

### 2. Melo sidecar audio wrapping changed earlier in the current worktree

This file changed actual TTS audio packaging:

- `local_tts_servers/melotts_http_server.py`

Old behavior:

- used `audioop.tomono(...)`
- used `audioop.ratecv(...)`
- wrote a temporary `.wav` file to disk and read it back

New behavior:

- uses NumPy channel averaging for mono conversion
- uses `numpy.interp(...)` for resampling
- writes WAV bytes in memory with `io.BytesIO`

This is the most relevant recent change if the symptom is:

- browser connects normally
- text appears
- no audible speech comes out

### 3. Other recent changes

These were also added, but they are less likely to directly cause silence:

- `src/blink/transports/smallwebrtc/transport.py`
  Warning throttling for repeated idle mic/camera stalls.
- `tests/test_browser_e2e.py`
  New opt-in browser smoke coverage.
- `pyproject.toml`
  Added Playwright dev dependency and test marker.
- `docs/*`
  Documentation updates.

## Fastest runtime rollback

If you want the closest runtime behavior to the previous browser setup without
editing code, use the old STT model explicitly:

```bash
pkill -f "blink.cli.local_browser|run-blink-browser.sh|run-local-browser.sh" || true
BLINK_LOCAL_STT_MODEL=mlx-community/whisper-medium-mlx ./scripts/run-blink-browser.sh
```

This rolls back only the STT default change.

## Fastest audio isolation rollback

If the symptom is specifically "no sound at all", bypass the Melo sidecar and
use the simpler Kokoro path to confirm whether the regression is in Melo or in
the browser/output path:

```bash
pkill -f "blink.cli.local_browser|run-blink-browser.sh|run-local-browser.sh" || true
BLINK_LOCAL_TTS_BACKEND=kokoro \
BLINK_LOCAL_STT_MODEL=mlx-community/whisper-medium-mlx \
./scripts/run-blink-browser.sh
```

Interpretation:

- If Kokoro speaks normally, the likely regression is in the Melo sidecar path.
- If Kokoro is also silent, the issue is probably not the recent Melo WAV
  packaging change.

## Source rollback target if you want the older Melo behavior back

If you want to restore the older Melo sidecar behavior in source code, revert
`local_tts_servers/melotts_http_server.py` so `wrap_pcm_as_wav(...)` goes back
to the previous implementation:

- use `audioop.tomono(...)` for stereo to mono conversion
- use `audioop.ratecv(...)` for resampling
- write/read through a temporary `.wav` file

That is the narrowest code rollback that restores the older TTS byte path.

## Source rollback target if you want the older STT defaults back

If you want to restore the previous Chinese STT default in source code, revert
these points:

- in `src/blink/cli/local_common.py`
  - set `DEFAULT_LOCAL_STT_MODEL = "mlx-community/whisper-medium-mlx"`
  - remove the language-aware `default_local_stt_model(...)` split
- in:
  - `src/blink/cli/local_voice.py`
  - `src/blink/cli/local_browser.py`
  - `src/blink/cli/local_prefetch.py`
  - `src/blink/cli/local_doctor.py`
  restore the previous `stt_model` resolution logic so it falls back to the old
  shared default

## Recommended rollback order

1. Try the runtime-only STT rollback first.
2. If silence remains, try the Kokoro isolation run.
3. If Kokoro works and Melo does not, revert the Melo sidecar WAV conversion
   path in `local_tts_servers/melotts_http_server.py`.
4. If both are silent, inspect the browser audio/output path instead of STT.

## What this document does not claim

This note does not claim the STT default change caused the current silence.
It records what changed and which rollback path is most rational for each
symptom.

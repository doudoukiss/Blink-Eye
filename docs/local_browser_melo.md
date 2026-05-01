# Local Browser Primary Paths

Blink has two primary browser/WebRTC voice paths. Treat them as equal main
product lanes:

- `browser-zh-melo`: Chinese browser voice with MeloTTS through
  `local-http-wav`, camera enabled by default.
- `browser-en-kokoro`: English browser voice with Kokoro, camera enabled by
  default, and no MeloTTS sidecar.

Run the Chinese Melo path:

```bash
./scripts/run-local-browser-melo.sh
```

Run the English Kokoro path:

```bash
./scripts/run-local-browser-kokoro-en.sh
```

For either path, open:

```text
http://127.0.0.1:7860/client/
```

Both launchers run browser/WebRTC voice and use protected playback by default.
The Melo launcher reuses or supervises the MeloTTS sidecar. The English Kokoro
launcher intentionally does not start MeloTTS and keeps browser vision available
unless you pass `--no-vision` or set `BLINK_LOCAL_BROWSER_VISION=0`.

Expected Chinese readiness output includes:

```text
runtime=browser transport=WebRTC profile=browser-zh-melo language=zh tts=local-http-wav/MeloTTS vision=on continuous_perception=off protected_playback=on barge_in=off barge_in_policy=protected client=http://127.0.0.1:7860/client/
```

Expected English readiness output includes:

```text
runtime=browser transport=WebRTC profile=browser-en-kokoro language=en tts=kokoro/English vision=on continuous_perception=off protected_playback=on barge_in=off barge_in_policy=protected client=http://127.0.0.1:7860/client/
```

Protected playback remains the safe default. Barge-in is enabled only when
`--allow-barge-in` is passed or `BLINK_LOCAL_ALLOW_BARGE_IN=1` is set.

Native English Kokoro launchers remain backend isolation lanes for mic, STT,
LLM, and TTS debugging. They are not the browser product UX path.

# 05 — Debugging Playbook: Browser/Melo Primary, Native Isolation Secondary

## Daily-use product run

```bash
./scripts/run-local-browser-melo.sh
```

Open:

```text
http://127.0.0.1:7860/client/
```

## Expected startup facts

The runtime should clearly print:

```text
runtime=browser
transport=WebRTC
tts=local-http-wav/MeloTTS
vision=on/off
continuous_perception=on/off
barge_in=off unless explicitly enabled
client=http://127.0.0.1:7860/client/
```

## If Blink cannot hear you

Check browser mic permission, `/api/runtime/client-media`, WebRTC connection state, STT backend logs, and `BrainVoiceInputHealthProcessor` state.

## If Blink cannot see camera

Check browser camera permission, client media mode, `LatestCameraFrameBuffer` sequence, camera frame age, and `CameraFeedHealthManager` reason codes.

## If Melo is silent

Check the MeloTTS sidecar health endpoint, local-http-wav base URL, `run-local-browser-melo.sh` supervisor logs, TTS queue events, and browser autoplay restrictions.

## If interruption fails

First confirm whether protected playback is expected. If `--allow-barge-in` is active, inspect echo-safe state, user speech events during assistant speech, interruption candidate/accepted/rejected events, and stale Melo chunk drops.

## If native path is involved

Treat native as isolation only. Do not chase product UX issues there unless the goal is specifically to isolate STT/LLM/TTS backends.

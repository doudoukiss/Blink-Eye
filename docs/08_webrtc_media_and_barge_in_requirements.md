# 08 — WebRTC Media and Barge-In Requirements

## Default mode

Protected playback remains the daily default. The user should not expect live interruption unless barge-in is explicitly enabled and echo safety is acceptable.

## Adaptive mode

When `--allow-barge-in` is enabled, the browser path should use all available signals:

```text
browser media constraints: echoCancellation, noiseSuppression, autoGainControl
assistant speaking flag
VAD during assistant speech
STT partial/final text
Melo output queue state
RTCPeerConnection/client health if available
user configuration: headphones/speakers
```

## Event sequence

```text
assistant_speaking
user_speech_during_assistant
interruption_candidate
classification: backchannel | real_interruption | echo_leak | noise
interruption_accepted or interruption_rejected
melo_queue_flush if accepted
listening_resumed
```

## UI state

```text
Protected playback
Barge-in armed
Interruption candidate
Interrupted — listening
Backchannel ignored
Echo risk — protected
```

## Regression target

No product path may self-interrupt on laptop speakers by default.

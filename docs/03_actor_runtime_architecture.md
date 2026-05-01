# 03 — Target actor runtime architecture

## Components

### ActorEventLedger

A durable, public-safe event stream. It should be used by UI, logs, evals, replay, and future avatar adapters.

```python
ActorEvent(
    event_id: int,
    session_id: str | None,
    client_id: str | None,
    turn_id: str | None,
    type: str,
    mode: str,
    floor_state: str,
    source: str,
    timestamp: str,
    metadata: dict,
    reason_codes: list[str],
)
```

### ConversationFloorController

Owns the conversational floor:

```text
idle
user_has_floor
assistant_has_floor
overlap
handoff
repair
```

It consumes VAD/STT/TTS/interruption/camera events and emits floor decisions.

### ActiveListener

Makes listening a visible behavior. It tracks speech duration, partial/final transcripts, topic/constraint hints, uncertainty, and readiness.

### SpeechPerformanceDirector

Turns assistant text into performable Melo speech chunks with subtitles, pauses, queue limits, and interruption-aware generation tokens.

### CameraSceneState

Tracks browser camera availability, freshness, low-rate scene/social signals, and on-demand vision grounding. It must distinguish “camera available” from “vision used for this answer.”

### PersonaPerformanceCompiler

Compiles persona references, memory traces, current modality, floor state, camera state, and user intent into a turn-level plan:

```python
PerformancePlanV2(
    stance,
    response_shape,
    memory_callbacks,
    persona_references,
    floor_policy,
    speech_policy,
    camera_policy,
    interruption_policy,
    forbidden_moves,
)
```

### BrowserActorSurface

The user-facing browser surface. It should prioritize five stable regions:

```text
Current state
Heard
Blink is saying
Camera / vision
Memory / persona used
```

Advanced logs and metrics remain available but should not dominate the main UX.

## Data flow

```text
WebRTC mic/camera
  -> local_browser.py frame observers
  -> ActorEventLedger
  -> FloorController + ActiveListener + CameraSceneState
  -> BrainRuntime / LLM / tools
  -> PersonaPerformanceCompiler
  -> SpeechPerformanceDirector
  -> MeloTTS HTTP-WAV sidecar
  -> BrowserActorSurface
  -> Actor trace + ActorBench
```

## Design constraints

- Default mode: browser/Melo, protected playback, camera on-demand.
- Adaptive barge-in only when echo-safe.
- UI must be useful even when STT has no partial transcripts.
- Event metadata must be bounded and public-safe by default.
- Future avatar adapters must consume actor events; they must not own the core conversation loop.

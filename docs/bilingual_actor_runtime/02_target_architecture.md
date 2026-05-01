# Target architecture

## Runtime paths

```text
Chinese path:
Browser/WebRTC mic+camera
  → browser-zh-melo profile
  → STT
  → Active Listener v2
  → Conversation Floor Controller
  → Memory/Persona retrieval
  → Moondream on-demand camera grounding
  → LLM
  → Performance Compiler v2
  → Melo Speech Performance Director
  → Browser audio + Actor Surface UI

English path:
Browser/WebRTC mic+camera
  → browser-en-kokoro profile
  → STT
  → Active Listener v2
  → Conversation Floor Controller
  → Memory/Persona retrieval
  → Moondream on-demand camera grounding
  → LLM
  → Performance Compiler v2
  → Kokoro Speech Performance Director
  → Browser audio + Actor Surface UI
```

## Core objects

```text
ActorEventV2
BrowserActorStateV2
ConversationFloorState
WebRTCAudioHealthV2
SpeechPerformanceChunk
ActiveListenerStateV2
CameraSceneState
PersonaReference
PerformancePlanV2
MemoryContinuityTrace
ActorBenchResult
ReleaseGateResult
```

## Control rules

- The two primary paths must share schemas, state APIs, event vocabulary, and evaluation gates.
- Profile-specific behavior should live in runtime profile data, capability maps, phrase lists, and TTS adapters—not in duplicated pipelines.
- Persistent traces are public-safe by default.
- Camera use must be explicit and honest: available, looking, used frame, stale, degraded, or disabled.
- Memory/persona must become behaviorally visible through the performance plan and UI trace.

# Target architecture

```text
Browser/WebRTC mic + camera
  -> actor event bus v2
  -> performance episode ledger v3
  -> actor control-frame scheduler
      -> conversation floor v3
      -> semantic active listener v3
      -> scene-social perception v2
      -> memory continuity v3
      -> persona reference selector
      -> performance planner v3
      -> dual TTS speech director v3
      -> browser actor surface / workbench
      -> bilingual performance bench v3
```

## Key runtime objects

### PerformanceEpisodeV3
A privacy-safe record of what happened in an interaction segment. It stores bounded text only when explicitly allowed and otherwise stores IDs, hashes, counts, labels, reason codes, durations, and evaluation features.

### ActorControlFrameV3
A boundary-aligned control packet that tells downstream components how to perform the next moment: floor policy, listening state, camera policy, speech chunk policy, memory/persona references, subtitle policy, and repair/interruption policy.

### SemanticListenerStateV3
The active listening state visible to the user: current topic, intent, constraints, uncertainty, readiness, language, and safe live summaries.

### SceneSocialStateV2
The honest camera state: permission, frame age, whether a frame was used, whether the user appears present, object-showing likelihood, last Moondream grounding, confidence, and failure/degradation reason.

### PerformancePlanV3
The response-performance plan: stance, shape, voice pacing, chunk budget, subtitle policy, camera reference policy, memory callback policy, persona anchors, repair stance, public UI status copy, and reason trace. It is additive to the existing memory/persona performance surface and consumes ActorControlFrameV3, SemanticListenerStateV3, SceneSocialStateV2, memory continuity, persona references, and conservative TTS capability declarations. It may adjust supported chunking/subtitle behavior, but it must not claim unsupported MeloTTS or Kokoro controls such as emotional prosody, arbitrary speech rate, stream abort, hardware expression, or fake-human likeness.

### PerformancePreferencePair
A pairwise dogfooding/evaluation record comparing two candidate behaviors or two builds on felt-heard, state clarity, interruption, voice pacing, camera honesty, memory usefulness, persona consistency, enjoyment, and not-fake-human.

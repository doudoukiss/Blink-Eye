# Performance Plan v3 and ActorControl Compiler

Phase 07 builds on the existing bilingual browser actor runtime. It does not
replace `/api/runtime/actor-state`, `/api/runtime/actor-events`, compatibility
performance endpoints, actor traces, or `performance_plan_v2`.

`PerformancePlanV3` is nested beside V2 in the memory/persona performance
payload. It is compiled from public-safe runtime evidence:

- the latest internal `ActorControlFrameV3`;
- `SemanticListenerStateV3` intent, readiness, and chip labels;
- `SceneSocialStateV2` camera honesty and scene transitions;
- `MemoryContinuityTrace` counts/effects and V3 discourse episode refs;
- selected persona reference IDs;
- selected `PersonaReferenceBankV3` style-anchor IDs and situation labels;
- conservative TTS capability declarations.

The plan exposes bounded `ui_status_copy` and `plan_summary` strings plus
policy objects for stance, response shape, voice pacing, speech chunk budget,
subtitle timing, camera reference honesty, memory callback, interruption, and
repair. `persona_anchor_refs_v3` adds text-free selected style-anchor summaries:
anchor IDs, situation keys, stance/shape labels, counts, and reason codes. Those
policy objects contain labels, booleans, counts, IDs, bounded numbers, and
reason codes only.

## Runtime Behavior

The browser voice policy processor may use V3 only for supported controls:
chunk target, hard maximum chunk size, and maximum chunks per flush. The
ActorControl scheduler remains the low-level authority for boundary timing and
lookahead. LLM text never controls VAD/STT/TTS/camera/interruption boundaries.

Repair and accepted-interruption plans prefer short interruptible speech.
Fresh camera grounding allows visual acknowledgement only when a fresh frame
was actually used. Stale or unavailable camera state forces a no-visual-claim
plan. Memory callbacks are used only when the memory continuity trace reports a
grounded effect. Phase 10 extends that policy with selected memory IDs,
discourse episode IDs, category/effect labels, conflict/staleness labels, and
cross-language transfer counts so concise preferences, project constraints,
corrections, repeated frustrations, and stale memories change behavior without
exposing raw memory bodies.

## TTS Capability Boundary

Both primary browser paths remain first-class:

- `browser-zh-melo`: `local-http-wav/MeloTTS`
- `browser-en-kokoro`: `kokoro/English`

Both support chunk boundaries and interruption flush through the current local
browser integration. They do not expose arbitrary speech-rate control,
emotional prosody, pause timing, partial stream abort, interruption discard,
hardware expression, realistic human likeness, identity cloning, face
reenactment, or raw-media avatar control.

Native PyAudio paths remain backend isolation lanes, not product UX lanes.
Protected playback remains default and continuous perception remains off by
default.

## Persistent Safety

Plan summaries are finite-template public copy. Persistent events, episodes,
control-frame replay, and bench artifacts must not store raw audio, raw images,
SDP, ICE candidates, secrets, hidden prompts, full model messages, raw memory
bodies, raw transcripts, or unbounded text.

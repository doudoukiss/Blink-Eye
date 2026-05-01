# Phase 09: Persona Reference Bank, Cross-Language Anchors, And Performance Compilers

Phase 09 converts Blink's style from hidden prose into a deterministic public plan for each
assistant turn. The plan is additive: existing memory/persona performance payloads remain schema v1,
`performance_plan_v2` remains nested inside them for browser UI and diagnostics, and
`PerformancePlanV3` now cites selected public style anchors.

## Persona Reference Bank

The built-in bank lives in `blink.brain.persona.reference_bank` and contains matched Chinese and
English references for:

- interruption
- correction
- disagreement
- deep technical planning
- casual chat
- camera use
- memory callback
- uncertainty
- concise answer
- playful restraint

Each reference is structured as public data: locale, scenario, stance, response shape, forbidden
moves, example input, example output, and performance notes. Runtime-selected summaries expose only
the reference id, locale, scenario, stance, response shape, short notes, and reason codes. Example
input/output and forbidden-move details are for tests/docs, not default live state or traces.

## PerformancePlanV2

`compile_performance_plan_v2(...)` consumes bounded public inputs:

- runtime profile, language, modality, and TTS label
- memory-use trace counts and public memory kinds
- active-listening hint counts
- camera scene state and grounded-vision flags
- conversation floor state and interruption policy context
- optional user affect/intent labels
- behavior-control profile values

The compiler returns:

- stance and response shape
- memory callback policy
- camera-reference policy
- interruption policy
- speech chunking hints
- UI state hints
- compact `style_summary`
- selected persona reference summaries
- reason codes

The compiler is deterministic and locale-selective. Chinese and English references are equivalent in
scenario coverage but are authored separately, so the English path is not a direct translation of the
Chinese path and vice versa.

## Public Safety

Default payloads never include raw audio, raw images, SDP, ICE candidates, secrets, credentials,
hidden prompts, full messages, unbounded transcripts, raw memory records, or full persona prose.
Memory policy exposes counts and public display kinds. Camera policy reports whether vision was
actually grounded by a fresh single frame, stale/limited, disabled, permission-limited, or failed.

Blink may express continuity, memory, style, and presence. It must not invent a human biography,
claim identity cloning, or present itself as a realistic human avatar.

## Browser Parity

Both primary browser profiles use the same v2 shape:

- `browser-zh-melo`: language `zh`, TTS label `local-http-wav/MeloTTS`
- `browser-en-kokoro`: language `en`, TTS label `kokoro/English`

Profile-specific labels and locale-specific reference text differ, but the structural payload remains
the same so the browser client can render one compact “style/persona used this turn” summary without
profile-specific branching.

## Persona Reference Bank V3

`PersonaReferenceBankV3` adds inspectable, situation-keyed anchors on top of the existing V1/V2
reference bank. The canonical anchor keys are:

- `interruption_response`
- `correction_response`
- `deep_technical_planning`
- `casual_check_in`
- `visual_grounding`
- `uncertainty`
- `disagreement`
- `memory_callback`
- `playful_not_fake_human`

Each anchor has public Chinese and English examples, behavior constraints, negative examples,
stance labels, response-shape labels, and reason codes. Runtime-selected summaries are text-free:
they expose anchor IDs, situation keys, labels, counts, and reason codes only. Full anchor examples
are inspectable through the existing operator workbench payload, not hidden prompts or model
messages.

`PerformancePlanV3` keeps `persona_reference_ids` for compatibility and adds
`persona_anchor_refs_v3`. Retrieval is situation-driven: floor repair selects interruption/correction
anchors, fresh or limited camera state selects visual-grounding honesty, memory continuity selects
memory callback, planning intent selects technical planning, and casual/playful behavior stays bounded
by the non-fake-human anchor.

Native PyAudio paths remain backend isolation lanes. The primary product paths remain the browser
Chinese/Melo/Moondream and English/Kokoro/Moondream lanes with protected playback on and continuous
perception off by default.

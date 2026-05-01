# Conversation Floor Controller

Phase 04 makes turn ownership explicit for the two primary browser paths:

- `browser-zh-melo`: Chinese, MeloTTS through `local-http-wav/MeloTTS`,
  browser/WebRTC media, and Moondream vision.
- `browser-en-kokoro`: English, Kokoro, browser/WebRTC media, and Moondream
  vision.

The controller is deterministic. It does not ask the LLM to decide low-level
turn physics, and Phase 04 builds on the existing browser actor runtime rather
than replacing its endpoints.

## State

`ConversationFloorState` reports one public state:

- `user_has_floor`
- `assistant_has_floor`
- `overlap`
- `handoff`
- `repair`
- `unknown`

The payload also includes boolean convenience flags, profile, language,
protected-playback policy, whether barge-in is armed, safe last-input labels,
transition counts, timestamps, and reason codes.

The same payload keeps `schema_version: 1` for compatibility and adds
`floor_model_version: 3` plus one detailed sub-state:

- `user_holding_floor`
- `assistant_holding_floor`
- `overlap_candidate`
- `accepted_interrupt`
- `ignored_backchannel`
- `repair_requested`
- `handoff_pending`
- `handoff_complete`

V3 also exposes public-safe phrase metadata: `phrase_class`,
`phrase_confidence`, `phrase_confidence_bucket`, `yield_decision`,
`echo_risk`, `barge_in_state`, `tts_chunk_role`, and
`low_confidence_transcript`. It never exposes the matched phrase text.

## Inputs

The browser runtime feeds symbolic inputs from existing public lifecycle
signals:

- browser/WebRTC VAD user start, continuing, and stop
- STT interim and final transcription events
- LLM response start and end
- TTS speech start and stop
- interruption candidate, accepted, rejected, suppressed, and resumed decisions
- WebRTC echo-health policy labels and STT confidence when available

The controller may inspect bounded in-memory STT text for classification, but
it only exposes derived labels such as `backchannel`, `hesitation`,
`correction`, `explicit_interruption`, or `meaningful`.

## Backchannel And Interruption Policy

Short Chinese and English backchannels such as `嗯`, `对`, `ok`, `yeah`, and
`right` keep `assistant_has_floor` while assistant speech policy says continue.
Continuers such as `继续` and `go on` are classified separately and also do not
kill assistant speech.

Explicit interruption phrases such as `等一下`, `停一下`, `wait`, and `hold on`
enter repair/yield policy when browser barge-in is armed, playback is
unprotected, or the runtime marks the setup echo-safe/adaptive. With protected
playback or high echo risk, the overlap remains visible as public state and
reason codes, but assistant audio is not forced to stop.

Corrections such as `不对`, `我不是说`, `actually`, and `I mean` enter `repair`
so downstream actor UI and diagnostics can distinguish repair from ordinary
listening.

Low-confidence STT partials are labeled `low_confidence_transcript` and do not
yield the floor.

## Public API

`GET /api/runtime/actor-state` includes:

```json
{
  "conversation_floor": {
    "schema_version": 1,
    "state": "assistant_has_floor",
    "profile": "browser-en-kokoro",
    "language": "en",
    "floor_model_version": 3,
    "sub_state": "assistant_holding_floor",
    "assistant_has_floor": true,
    "last_input_type": "tts_started",
    "last_text_kind": "empty",
    "phrase_class": "empty",
    "yield_decision": "continue_assistant",
    "reason_codes": [
      "conversation_floor:v1",
      "conversation_floor:v3",
      "floor_state:assistant_has_floor"
    ]
  }
}
```

`GET /api/runtime/performance-state` remains schema v1.

## Actor Events

Floor transitions emit `floor.transition` v1 performance events and
`floor_transition` v2 actor events. Transition metadata is public-safe:

- `floor_state`
- `floor_sub_state`
- `previous_floor_state`
- `input_type`
- `text_kind`
- `phrase_class`
- `phrase_confidence_bucket`
- `yield_decision`
- `transition_count`
- bounded boolean policy flags

Default actor traces do not persist raw transcript text, raw audio, raw images,
SDP, ICE candidates, secrets, credentials, prompts, or full model messages.

Native PyAudio launchers remain useful backend isolation lanes for STT/TTS
debugging, but Phase 04 floor policy is product-targeted at the two
browser/WebRTC paths.

# Moondream Camera Scene State And Grounded Perception Parity

Phase 08 adds `CameraSceneState` as the public camera truth surface for the two
primary browser paths:

- `browser-zh-melo`: Chinese, MeloTTS through `local-http-wav`, Moondream
- `browser-en-kokoro`: English, Kokoro, Moondream

Both profiles keep browser vision enabled by default and route through the same
`/client/` browser UI. Continuous perception remains off unless explicitly
enabled with `--continuous-perception` or `BLINK_LOCAL_CONTINUOUS_PERCEPTION=1`.

## Public State

`GET /api/runtime/actor-state` exposes:

- top-level `camera_scene`
- `camera.scene`
- existing `camera.presence` for compatibility

`GET /api/runtime/performance-state` remains schema v1 and may include the same
`camera_scene` payload additively.

`CameraSceneState` reports `disabled`, `permission_needed`,
`waiting_for_frame`, `available`, `looking`, `stale`, `stalled`, or `error`.
It includes frame sequence, frame age, freshness, track state, on-demand vision
state, last-used frame metadata, grounding mode, degradation, and reason codes.

Phase 06 adds nested `scene_social_state_v2` without replacing the existing
schema-v1 camera scene payload. It reports camera permission/status, frame
freshness, user/face/body/hands/object hints, object-showing likelihood,
scene-change reason, the last Moondream result state, confidence, and the
camera honesty state. The honesty labels are:

- `can_see_now`: a fresh frame was used in the current answer.
- `recent_frame_available`: a recent frame exists, but the current answer did
  not use vision.
- `available_not_used`: camera/vision can be available, but no frame has been
  used for the answer.
- `unavailable`: camera or usable vision is disabled, stale, blocked, or
  otherwise unavailable.

## Grounding Policy

Blink may say it used the camera only when `current_answer_used_vision=true`.
That is set after a successful on-demand `fetch_user_image` call over a fresh
still frame. The state must otherwise say that visual context is unavailable,
waiting, stale, stalled, or limited.

Moondream remains on demand. Frame caching and camera health monitoring can run
for browser diagnostics, but no continuous VLM interpretation runs by default.
Low-frequency scene updates remain explicit opt-in via the existing continuous
perception switch; the two primary browser product paths keep continuous
perception off by default. Native PyAudio paths remain backend isolation lanes,
not the primary camera/vision UX lane.

## Event Mapping

`vision.fetch_user_image_requested` and `vision.fetch_user_image_start` put the
actor in `looking`. `vision.fetch_user_image_success` records
`grounding_mode=single_frame`. `vision.fetch_user_image_error` maps to an actor
`error`. Camera stale or stalled events map to `degraded`, and camera resume
events map to `recovered`.

Scene-social transitions are public labels only:

- `camera_ready`
- `looking_requested`
- `frame_captured`
- `vision_answered`
- `vision_stale`
- `vision_unavailable`

Performance episodes and actor control frames replay these labels, freshness,
honesty, object-showing likelihood, user-presence hints, and Moondream result
state without carrying raw images or raw Moondream text.

## Privacy Rules

Camera scene state stores only bounded public metadata. It must not expose raw
images, raw audio, SDP, ICE candidates, device labels, credentials, hidden
prompts, full user messages, or raw Moondream text in persistent traces.

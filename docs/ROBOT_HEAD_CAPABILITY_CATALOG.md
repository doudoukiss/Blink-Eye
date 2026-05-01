# Robot Head Capability Catalog

This document mirrors the Blink-owned semantic robot-head catalog used by the
controller and preview drivers.

## Semantic Units

- `head_turn`: full verified yaw envelope for proof lanes, with smaller public conversational use
- `neck_pitch`: operator-gated structural pitch for proof and show lanes
- `neck_tilt`: operator-gated structural tilt compiled against the family-specific isolated-tilt envelope
- `eye_yaw`: widened to the verified eye-yaw family envelope for proof lanes
- `eye_pitch`: widened to the verified eye-pitch family envelope for proof lanes
- `left_lids`
- `right_lids`
- `left_brow`
- `right_brow`

These are semantic units, not raw actuators. Blink's live driver now translates
them onto the packaged hardware profile mirror in
`src/blink/embodiment/robot_head/live_hardware_profile.json`, while keeping raw
servo IDs and counts behind the driver boundary.

## Persistent States

- `neutral`
- `friendly`
- `listen_attentively`
- `thinking`
- `focused_soft`
- `confused`
- `safe_idle`

Held states intentionally stay in the eye area. Structural motion is excluded
so conversational embodiment does not pin the head in risky physical poses.

The public held states were retuned to be visibly readable on hardware. In
particular, `friendly`, `listen_attentively`, and `thinking` are no longer the
small under-expressive eye-area writes from the first Blink pass.

## Public Motifs

- `acknowledge`
- `blink`
- `wink_left`
- `wink_right`
- `look_left`
- `look_right`

These public motifs now use the stronger `conversation_readable` preset instead
of the earlier timid conversational lane.

## Operator / Preview-Only Motifs

- `curious_tilt`
- `listen_engage`
- `thinking_shift`

`curious_tilt` remains preview-only because it is not part of the public tool
surface and it has not been promoted into the tracked operator proof ladder.
`listen_engage` and `thinking_shift` are policy-only cues used to make
speech-reactive embodiment visibly legible instead of reading like servo noise.

## Operator Proof And Show Motifs

- V3: `investor_head_turn_left_v3`, `investor_head_turn_right_v3`,
  `investor_head_sweep_left_v3`, `investor_head_sweep_right_v3`
- V4: `investor_eye_yaw_left_v4`, `investor_eye_yaw_right_v4`,
  `investor_eye_pitch_up_v4`, `investor_eye_pitch_down_v4`
- V5: `investor_both_lids_v5`, `investor_left_eye_lids_v5`,
  `investor_right_eye_lids_v5`, `investor_blink_v5`
- V6: `investor_brows_both_v6`, `investor_brow_left_v6`,
  `investor_brow_right_v6`
- V7: `investor_neck_tilt_left_v7`, `investor_neck_tilt_right_v7`,
  `investor_neck_pitch_up_v7`, `investor_neck_pitch_down_v7`
- V8: `attentive_notice_right`, `attentive_notice_left`,
  `guarded_close_right`, `guarded_close_left`, `curious_lift`,
  `reflective_lower`, `skeptical_tilt_right`, `empathetic_tilt_left`,
  `playful_peek_right`, `playful_peek_left`, `bright_reengage`,
  `doubtful_side_glance`

These motifs are intentionally `public=False`. They are operator-only motions,
not normal user-request tools.

## Preset Intent

- `conversation_safe`: default path for automatic embodiment and normal user
  commands
- `conversation_readable`: slower, larger conversational lane for public gaze,
  blinks, and policy-driven social responses
- `preview_safe`: motion that is valid only for preview-mode inspection until
  live validation catches up
- `operator_proof_safe`: V3-V6 family proof lane
- `operator_neck_safe`: slower V7 neck proof lane and neck-bearing V8 motifs
- `operator_expressive_v8`: expressive V8 lane with conservative `90/24`
  kinetics

# Robot Head Live Limits

This document is the checked-in human-facing numeric truth for the currently calibrated live robot head.

Packaged live profile mirror:
- `src/blink/embodiment/robot_head/live_hardware_profile.json`

Historical calibration source:
- the original live calibration was captured from a local `runtime/` artifact
  during hardware bring-up; runtime artifacts are intentionally not committed in
  the source-first repository.

Motion-policy companion:
- `docs/body_motion_tuning.md`

Last saved live-calibration update: `2026-04-10T11:41:53.627287+00:00`
Most recent historical live artifact directory:
`runtime/serial/live_range_revalidation/20260410T113342365139Z_live_joint_revalidation`
from the local hardware bench.

Conversion:
- STS3032 raw counts run from `0..4095` for one full turn.
- `1 count = 360 / 4096 = 0.087890625°`.
- Degree values below are servo-shaft equivalent angles, not guaranteed visible head-output angles after linkage geometry.

## Scope

- Applies to the connected 11-servo robot head using the saved live calibration above.
- Planner/runtime semantic behavior compiles through
  `src/blink/embodiment/robot_head/`; this doc records calibrated raw limits and
  direction semantics.
- If future live revalidation changes the saved calibration, update this doc from the same calibration file and revalidation artifact.

## Per-Joint Live Limits

| Joint | Raw Min | Neutral | Raw Max | Neutral→Low (cts/deg) | Neutral→High (cts/deg) | Total Span (cts/deg) |
|---|---:|---:|---:|---:|---:|---:|
| `head_yaw` | 1641 | 2096 | 2453 | 455 / 39.99 | 357 / 31.38 | 812 / 71.37 |
| `head_pitch_pair_a` | 1362 | 2041 | 2647 | 679 / 59.68 | 606 / 53.26 | 1285 / 112.94 |
| `head_pitch_pair_b` | 1447 | 2058 | 2681 | 611 / 53.70 | 623 / 54.76 | 1234 / 108.46 |
| `lower_lid_left` | 1948 | 2050 | 2346 | 102 / 8.96 | 296 / 26.02 | 398 / 34.98 |
| `upper_lid_left` | 1550 | 2048 | 2242 | 498 / 43.77 | 194 / 17.05 | 692 / 60.82 |
| `lower_lid_right` | 1749 | 2045 | 2140 | 296 / 26.02 | 95 / 8.35 | 391 / 34.37 |
| `upper_lid_right` | 1848 | 2006 | 2545 | 158 / 13.89 | 539 / 47.37 | 697 / 61.26 |
| `eye_pitch` | 1953 | 2147 | 2440 | 194 / 17.05 | 293 / 25.75 | 487 / 42.80 |
| `eye_yaw` | 1756 | 2073 | 2332 | 317 / 27.86 | 259 / 22.76 | 576 / 50.62 |
| `brow_left` | 1897 | 2094 | 2197 | 197 / 17.31 | 103 / 9.05 | 300 / 26.37 |
| `brow_right` | 1897 | 2000 | 2197 | 103 / 9.05 | 197 / 17.31 | 300 / 26.37 |

## Family Practical Summary

| Family | Low Direction | High Direction | Low Side (cts/deg) | High Side (cts/deg) | Total Span (cts/deg) |
|---|---|---|---:|---:|---:|
| `head_yaw` | left | right | 455 / 39.99 | 357 / 31.38 | 812 / 71.37 |
| `neck_pitch` | down | up | 623 / 54.76 | 606 / 53.26 | 1229 / 108.02 |
| `neck_tilt` | left | right | 611 / 53.70 | 606 / 53.26 | 1217 / 106.96 |
| `eye_yaw` | left | right | 317 / 27.86 | 259 / 22.76 | 576 / 50.62 |
| `eye_pitch` | down | up | 194 / 17.05 | 293 / 25.75 | 487 / 42.80 |
| `upper_lids` | close | open | 498 / 43.77 | 158 / 13.89 | 656 / 57.66 |
| `lower_lids` | close | open | 296 / 26.02 | 95 / 8.35 | 391 / 34.37 |
| `brows` | lower | raise | 197 / 17.31 | 103 / 9.05 | 300 / 26.37 |

## Direction Semantics

- `head_yaw`: raw `+` = right, raw `-` = left.
- `eye_yaw`: raw `+` = eyes right, raw `-` = eyes left.
- `eye_pitch`: raw `+` = eyes up, raw `-` = eyes down.
- `upper_lid_left`: raw `+` = open, raw `-` = close.
- `upper_lid_right`: raw `+` = close, raw `-` = open.
- `lower_lid_left`: raw `+` = close, raw `-` = open.
- `lower_lid_right`: raw `+` = open, raw `-` = close.
- `brow_left`: raw `+` = raise, raw `-` = lower.
- `brow_right`: raw `+` = lower, raw `-` = raise.

## Neck Pair Coupling

- `head_pitch_pair_a` raw `+` with `head_pitch_pair_b` raw `-` = head up.
- `head_pitch_pair_a` raw `-` with `head_pitch_pair_b` raw `+` = head down.
- `head_pitch_pair_a` raw `+` alone = tilt right.
- `head_pitch_pair_b` raw `-` alone = tilt left.
- The neck-pair neutral should stay near a level center. If either pitch servo becomes suspicious in `usable-range`, recenter the pair before trusting any bold pitch demo.

## Validation Method

- Live revalidation is bench-only and operator-confirmed.
- The workflow starts from a saved non-template live calibration and an active arm lease.
- The neck pair is first recentered to a level neutral and saved from live readback.
- Each joint family is then stepped outward with real dwell and readback until the first failing condition, and the last confirmed passing readback becomes the practical saved limit.
- Failing conditions include serial instability, servo error bits, repeated abnormal load, non-convergence, or operator abort on visible strain.

## Caveats

- These values are for this assembled head and its saved live calibration, not a universal hardware template.
- The checked-in doc is the human-facing numeric truth; the packaged
  `live_hardware_profile.json` is the source-tree executable mirror.
- The `neck_tilt` family row above is derived from the current saved neck-pair bounds. Isolated live tilt on this head is narrower than that geometric summary; use the latest family-by-family revalidation findings before treating `neck_tilt` as a bold demo envelope.
- This document mirrors the saved live calibration snapshot. If a fresh family-by-family revalidation session stops early, keep the generated artifacts with the saved file before treating every limit as fully revalidated.
- If the hardware is absent or live transport is unavailable, the rest of Blink
  remains usable through bodyless, preview, and simulation paths.

# Robot Head Live Revalidation Findings (2026-04-10)

This note records the live family-by-family hardware work done on the connected
11-servo head on the Mac serial path. Use it alongside
[robot_head_live_limits.md](./robot_head_live_limits.md). Artifact paths below
refer to local hardware-bench outputs from that session; those runtime outputs
are intentionally not committed in the source-first repository.

## What Was Confirmed Live

- Transport: `live_serial`
- Port: `/dev/cu.usbmodemXXXX`
- Baud: `1000000`
- Verified motion-control settings on live hardware:
  - safe speed `120`
  - safe acceleration `40`
  - both applied and read back successfully on the Mac path

## Family Results

### Fully revalidated in this pass

- `head_yaw`
  - artifact: `runtime/serial/live_range_revalidation/20260410T104400408422Z_live_joint_revalidation`
  - saved result: `1641 / 2096 / 2453`

- `eye_yaw`
  - artifact: `runtime/serial/live_range_revalidation/20260410T112810173162Z_live_joint_revalidation`
  - saved result: `1756 / 2073 / 2332`

- `eye_pitch`
  - artifact: `runtime/serial/live_range_revalidation/20260410T112847883232Z_live_joint_revalidation`
  - saved result: `1953 / 2147 / 2440`

- `upper_lids`
  - artifact: `runtime/serial/live_range_revalidation/20260410T112917261338Z_live_joint_revalidation`
  - saved results:
    - `upper_lid_left = 1550 / 2048 / 2242`
    - `upper_lid_right = 1848 / 2006 / 2545`

- `lower_lids`
  - artifact: `runtime/serial/live_range_revalidation/20260410T113057210248Z_live_joint_revalidation`
  - saved results:
    - `lower_lid_left = 1948 / 2050 / 2346`
    - `lower_lid_right = 1749 / 2045 / 2140`

- `brows`
  - artifact: `runtime/serial/live_range_revalidation/20260410T113342365139Z_live_joint_revalidation`
  - saved results:
    - `brow_left = 1897 / 2094 / 2197`
    - `brow_right = 1897 / 2000 / 2197`

### Confirmed as family-specific only

- `neck_tilt`
  - artifact: `runtime/serial/live_range_revalidation/20260410T113751305410Z_live_joint_revalidation`
  - isolated live tilt is much narrower than the full neck-pair raw bounds
  - confirmed usable isolated-tilt envelope:
    - left tilt via `head_pitch_pair_b raw -`: last confirmed passing readback `2009`
    - right tilt via `head_pitch_pair_a raw +`: last confirmed passing readback `2251`
  - this is a family-specific usable range, not a replacement for the global raw hard bounds of servos 2 and 3

## Neck Pair Status

- Neck neutrals were recentered repeatedly during this pass.
- Current saved live neutrals after the last clean recenter:
  - `head_pitch_pair_a.neutral = 2041`
  - `head_pitch_pair_b.neutral = 2058`

- Global saved hard bounds currently remain:
  - `head_pitch_pair_a = 1362 / 2041 / 2647`
  - `head_pitch_pair_b = 1447 / 2058 / 2681`

## What Remains Partially Confirmed

- `neck_pitch`
  - the direct coupled-neck family still stalls during the neutral reset between directions
  - latest failed artifact: `runtime/serial/live_range_revalidation/20260410T114153062038Z_live_joint_revalidation`
  - meaning:
    - the neck pair can be centered and moved live
    - the current automation still needs another recovery-tuned pass before the full paired up/down family is considered freshly revalidated end-to-end on this head

## Practical Interpretation

- For investor/demo work today:
  - trust the saved yaw, eye, lid, and brow values in the canonical live-limits doc
  - trust the neck-pair neutral recenter values above
  - treat isolated `neck_tilt` as a narrower family-specific envelope than the global servo raw bounds imply
  - do not treat `neck_pitch` as freshly revalidated for bold end-stop demos until the reset-stall issue is fully cleared

## Why This Note Exists

The current calibration model stores per-servo raw hard bounds, while the neck pair also has family-specific usable ranges for coupled pitch and isolated tilt. This note keeps that distinction explicit so the checked-in numeric truth stays honest.

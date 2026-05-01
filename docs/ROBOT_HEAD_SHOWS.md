# Robot Head Proof Shows

Blink now carries its own operator proof-show lane for the connected robot
head. These shows are implemented inside Blink and run through the same
controller and live driver boundary as the normal embodiment stack.

For the full operator command boundary and a strict mapping from request to
motion family, see
[`ROBOT_HEAD_OPERATOR_HANDBOOK.md`](./ROBOT_HEAD_OPERATOR_HANDBOOK.md).

## Show Catalog

- `investor_head_motion_v3` / `v3`: head-yaw proof with max turns plus strong left/right sweeps
- `investor_eye_motion_v4` / `v4`: eye yaw and pitch proof
- `investor_lid_motion_v5` / `v5`: lid and blink proof
- `investor_brow_motion_v6` / `v6`: brow proof
- `investor_neck_motion_v7` / `v7`: protective neck pitch and tilt proof
- `investor_expressive_motion_v8` / `v8`: twelve-motif expressive ladder

## CLI

List the available shows:

```bash
blink-local-robot-head-show --list-shows
```

Run a preview-only proof:

```bash
blink-local-robot-head-show v3 --robot-head-driver preview
```

Run a live family-safe proof:

```bash
blink-local-robot-head-show v3 --robot-head-driver live --robot-head-live-arm
```

Run a live sensitive proof:

```bash
blink-local-robot-head-show v8 \
  --robot-head-driver live \
  --robot-head-live-arm \
  --allow-sensitive-motion
```

## Safety Rules

- `v7` and `v8` are blocked unless `--allow-sensitive-motion` is set.
- `live` mode still requires `--robot-head-live-arm`. Without it, Blink records
  preview traces instead of moving hardware.
- Every show starts from neutral and ends at neutral.
- JSON reports are written under `artifacts/robot_head_shows/`.
- Preview traces are written under `artifacts/robot_head_preview/`.

## Kinetics

- `conversation_safe`: normal conversational embodiment
- `conversation_readable`: larger and slower public conversational lane
- `operator_proof_safe`: V3-V6 proof lane, conservative `100/32`
- `operator_neck_safe`: slower V7 neck lane
- `operator_expressive_v8`: V8 expressive lane, conservative `90/24`

These are Blink-owned kinetics choices. They are not runtime imports from an
external robotics codebase.

## Current Tuning Intent

- V3-V6 are intentionally larger than the first Blink implementation, because
  the earlier proof cues were physically too small to read clearly.
- V7 uses the family-specific isolated-tilt envelope from the checked-in live
  docs instead of the full raw neck-pair span.
- V8 stays staged and deliberate: structural motion sets first, expressive
  units stack after it, then each expressive group releases before neutral
  return.

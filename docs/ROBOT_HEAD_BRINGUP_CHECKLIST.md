# Robot Head Bring-Up Checklist

This checklist defines the clean handoff from software prep to the first real
Blink-controlled live session on the connected head.

## Software Gate

- Run the pure-python robot-head tests.
- Run the local workflow tests that cover `--robot-head-driver`.
- Start `blink-local-voice --robot-head-driver preview`.
- Start `blink-local-robot-head-show --list-shows`.
- Start `blink-local-robot-head-show v3 --robot-head-driver preview`.
- Exercise the preview lane with scripted requests:
  - "listen"
  - "think"
  - "blink"
  - "look left"
  - "return to neutral"
- Review the generated trace artifacts under `artifacts/robot_head_preview/`.
- Review the generated proof-show reports under `artifacts/robot_head_shows/`.

## Read-Only Live Probe

- Install the serial dependency lane:

```bash
uv sync --python 3.12 --group dev --extra robot-head
```

- Start with live motion unarmed:

```bash
blink-local-voice --robot-head-driver live --robot-head-port /dev/cu.usbmodemXXXX
```

- Confirm the serial path is owned by only one process.
- Verify status reporting, ownership reporting, servo presence, and unarmed
  preview-fallback behavior.
- Confirm `blink-local-robot-head-show v3 --robot-head-driver live` returns
  preview fallback rather than moving the head while unarmed.

## First Armed Session

- Arm live motion explicitly:

```bash
blink-local-voice \
  --robot-head-driver live \
  --robot-head-port /dev/cu.usbmodemXXXX \
  --robot-head-live-arm
```

- Let Blink recover to neutral before any family motion proof.
- Run only family-safe proof actions first:
  - neutral
  - blink
  - left/right look
- Then run the Blink-owned family proof ladder:
  - `blink-local-robot-head-show v3 --robot-head-driver live --robot-head-live-arm`
  - `blink-local-robot-head-show v4 --robot-head-driver live --robot-head-live-arm`
  - `blink-local-robot-head-show v5 --robot-head-driver live --robot-head-live-arm`
  - `blink-local-robot-head-show v6 --robot-head-driver live --robot-head-live-arm`
- Do not enable autonomous conversational embodiment until those proofs are
  stable.

## First Live Validation Goals

- Confirm that semantic neutral maps to a true mechanical neutral.
- Validate head-turn motifs against real readback and visible behavior.
- Check that the live status payload reports stable voltage, temperature, and
  servo-status bytes across the proof lane.
- Validate V7 and V8 only after the family-safe ladder is clean:
  - `blink-local-robot-head-show v7 --robot-head-driver live --robot-head-live-arm --allow-sensitive-motion`
  - `blink-local-robot-head-show v8 --robot-head-driver live --robot-head-live-arm --allow-sensitive-motion`
- Keep neck motion in the operator-only show lane until Blink has broader live
  evidence and stronger recovery heuristics.
- Record any range or asymmetry findings in tracked Blink docs, not only in
  local notes.

## Expansion Gate

Only after the family-safe proof lane is clean should the next live phase add:

- live structural motifs beyond conservative head-turn
- non-preview neck support
- richer automatic embodiment policy
- browser/WebRTC reuse of the same controller and driver boundary

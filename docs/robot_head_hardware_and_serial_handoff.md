# Robot Head Hardware And Serial Handoff

This document is the public handoff for Blink's optional 11-servo robot-head
path. It explains the hardware model, the Mac serial connection, the current
Blink-owned software boundary, and the operator workflow without depending on
external robotics code or local runtime artifacts.

For the product-level integration view, read
[ROBOT_HEAD_INTEGRATION.md](./ROBOT_HEAD_INTEGRATION.md). For exact operator
commands, read
[ROBOT_HEAD_OPERATOR_HANDBOOK.md](./ROBOT_HEAD_OPERATOR_HANDBOOK.md).

## Scope

Blink treats physical embodiment as optional. The main framework remains useful
through text, browser/WebRTC voice, local voice, preview robot-head traces, and
deterministic simulation even when no hardware is attached.

This handoff covers the current Mac-first serial path for the real 11-servo
head. It does not try to replace the executable source files or the live limits
table:

- [robot_head_live_limits.md](./robot_head_live_limits.md)
- [robot_head_live_revalidation_2026-04-10.md](./robot_head_live_revalidation_2026-04-10.md)
- [src/blink/embodiment/robot_head/live_hardware_profile.json](../src/blink/embodiment/robot_head/live_hardware_profile.json)

## Hardware Summary

The current physical target is an 11-servo Feetech/STS3032-style head driven
from a Mac over a serial bus.

Active joints:

- `head_yaw`
- `head_pitch_pair_a`
- `head_pitch_pair_b`
- `eye_yaw`
- `eye_pitch`
- `upper_lid_left`
- `upper_lid_right`
- `lower_lid_left`
- `lower_lid_right`
- `brow_left`
- `brow_right`

The neck uses a coupled pair. `head_pitch_pair_a` and
`head_pitch_pair_b` together produce pitch, and the same pair participates in
tilt. That is why Blink distinguishes raw servo limits from family-specific
usable motion envelopes.

## Maintained Connection Path

The maintained path is Mac-first:

1. Blink owns the local runtime, robot-head controller, serial protocol, and
   proof-show CLI.
2. The Mac opens one serial device connected to the servo bus.
3. The live driver speaks Feetech packets directly from Blink-owned code.
4. Semantic motion compiles through the controller and hardware profile before
   any raw servo target is written.

Current defaults:

- transport: live serial
- baud: `1000000`
- typical port shape: `/dev/cu.usbmodemXXXX`
- serial ownership: single process at a time

Only one process should own the serial port. If a runtime, proof-show, doctor,
or another serial tool already holds `/dev/cu.*`, later commands can fail even
when the hardware is healthy.

## Source Boundaries

Primary implementation:

- [serial_protocol.py](../src/blink/embodiment/robot_head/serial_protocol.py)
- [live_driver.py](../src/blink/embodiment/robot_head/live_driver.py)
- [live_hardware.py](../src/blink/embodiment/robot_head/live_hardware.py)
- [controller.py](../src/blink/embodiment/robot_head/controller.py)
- [catalog.py](../src/blink/embodiment/robot_head/catalog.py)
- [policy.py](../src/blink/embodiment/robot_head/policy.py)
- [simulation.py](../src/blink/embodiment/robot_head/simulation.py)
- [show.py](../src/blink/embodiment/robot_head/show.py)

Supporting docs:

- [ROBOT_HEAD_CAPABILITY_CATALOG.md](./ROBOT_HEAD_CAPABILITY_CATALOG.md)
- [ROBOT_HEAD_BRINGUP_CHECKLIST.md](./ROBOT_HEAD_BRINGUP_CHECKLIST.md)
- [ROBOT_HEAD_SHOWS.md](./ROBOT_HEAD_SHOWS.md)

Test coverage:

- [test_robot_head_live_driver.py](../tests/test_robot_head_live_driver.py)
- [test_robot_head_controller.py](../tests/test_robot_head_controller.py)
- [test_robot_head_simulation.py](../tests/test_robot_head_simulation.py)
- [test_robot_head_tools.py](../tests/test_robot_head_tools.py)
- [test_robot_head_shows.py](../tests/test_robot_head_shows.py)

## Feetech Packet Model

The live wire protocol uses Feetech-style framed serial packets:

- header: `0xFF 0xFF`
- broadcast ID: `0xFE`
- checksum: inverted byte-sum over the packet body

Blink implements the packet builder/parser in
[serial_protocol.py](../src/blink/embodiment/robot_head/serial_protocol.py).
The live driver uses ping, read, write, sync-read, and sync-write patterns for
presence checks, telemetry, and synchronized movement.

Important telemetry includes present position, speed, load, voltage,
temperature, status, moving state, and current. These values stay behind the
driver/status boundary; planner and LLM-facing surfaces receive bounded semantic
status, not raw bus control.

## Safety Model

Blink's robot-head path is deliberately layered:

- Planner-facing requests use finite semantic actions.
- Raw servo IDs and counts stay inside the live driver and hardware profile.
- Daily-use browser/voice paths expose only bounded actions such as blink, wink,
  look left/right, neutral return, and status.
- Neck pitch and tilt remain operator/proof-show only.
- Live motion requires explicit arming.
- Unarmed live mode can still probe status read-only and fall back to preview
  traces instead of claiming motion succeeded.

Normal live motion should begin with preview or simulation, then read-only live
probe, then an explicitly armed session. The first real movement should be
neutral recovery, blink, and left/right look before any expressive proof show.

## Runtime Commands

Preview or simulation development:

```bash
blink-local-voice --robot-head-driver preview
blink-local-browser --robot-head-driver simulation --vision
blink-local-robot-head-show v3 --robot-head-driver preview
```

Read-only live probe:

```bash
blink-local-voice \
  --robot-head-driver live \
  --robot-head-port /dev/cu.usbmodemXXXX
```

Armed live session:

```bash
blink-local-voice \
  --robot-head-driver live \
  --robot-head-port /dev/cu.usbmodemXXXX \
  --robot-head-live-arm
```

Proof-show ladder:

```bash
blink-local-robot-head-show --list-shows
blink-local-robot-head-show v3 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v4 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v5 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v6 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v7 --robot-head-driver live --robot-head-live-arm --allow-sensitive-motion
blink-local-robot-head-show v8 --robot-head-driver live --robot-head-live-arm --allow-sensitive-motion
```

## Motion Families

The daily-use surface intentionally stays smaller than the hardware:

- daily-use: blink, wink, look left, look right, neutral return, status
- automatic policy: listen, think, speak-friendly, safe-idle
- operator mode: named public states and motifs
- proof-show mode: V3-V8 family and expressive show lanes

This keeps normal conversational embodiment readable without exposing raw
calibration, serial, or neck-control details to user prompts.

## Runtime Outputs

Generated traces and reports are disposable local artifacts:

- `artifacts/robot_head_preview/`
- `artifacts/robot_head_simulation/`
- `artifacts/robot_head_shows/`
- `runtime/robot_head/`

Those directories are intentionally ignored in the source-first repository. They
are useful for local debugging and hardware evidence, but they should not be
published as part of the core source tree.

## Handoff Summary

Blink's robot-head path is now source-contained: semantic catalog, controller,
policy, simulation, proof shows, serial protocol, and live driver all live under
`src/blink/embodiment/robot_head/`. The public runtime stays safe by default,
simulation is the preferred development and CI backend, and live hardware motion
requires explicit operator intent.

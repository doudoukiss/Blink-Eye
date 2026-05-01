# Robot Head Integration

Blink hosts a Blink-owned robot-head embodiment layer without taking a
dependency on an external robotics codebase. The head is treated as optional
physical embodiment while Blink remains the intelligence, decision, and
orchestration layer.

For the strict operator command surface, supported daily-use commands, and
V3-V8 motion decomposition, see
[`ROBOT_HEAD_OPERATOR_HANDBOOK.md`](./ROBOT_HEAD_OPERATOR_HANDBOOK.md).

## Architecture

- `RobotHeadCapabilityCatalog` defines planner-facing semantic units, persistent
  states, and motifs.
- `RobotHeadController` is the single serialized command owner. Both automatic
  embodiment policy and explicit LLM tools route through it.
- `RobotHeadDriver` is the hardware boundary. `mock` and `preview` remain the
  deterministic test lanes. `live` now owns the real Feetech serial path inside
  Blink itself.
- `RobotHeadLiveDriverConfig` and the packaged
  `live_hardware_profile.json` provide Blink-owned serial defaults for the
  current 11-servo head: servo IDs, live neutrals, raw limits, port hint, and
  conservative kinetics.
- `EmbodimentPolicyProcessor` reacts to Blink frame events and issues readable
  discrete actions:
  - `UserStartedSpeakingFrame` -> `listen_engage`, then `listen_attentively`
  - `UserStoppedSpeakingFrame` -> `thinking_shift`, then `thinking`
  - `BotStartedSpeakingFrame` -> `acknowledge`, then `friendly`
  - idle / cancel / stop / error -> `safe_idle`, then `neutral`
  - repeated turn-start noise is deduplicated so Blink does not keep rewriting
    the same tiny pose while the user is already speaking

## Safety Model

- Planner-facing values are semantic and normalized. They are not raw servo
  counts and do not expose serial-level controls to the LLM.
- Persistent states are eye-area-only held configurations.
- Structural motion lives in motifs, not in held states.
- Neck pitch and tilt are excluded from the public conversational tool surface.
  They are available only in explicit operator proof and show lanes.
- Driver status is structured and explicit about preview fallback, ownership,
  and degraded or unarmed states.
- Live motion requires both:
  - Blink-side serial ownership via a lock file under `runtime/robot_head/`
  - explicit arming via `--robot-head-live-arm` or
    `BLINK_LOCAL_ROBOT_HEAD_ARM=1`
- If the port is absent, ownership is busy, the arm lease is invalid, or a
  command touches preview-only motion, Blink records a preview trace instead of
  pretending that hardware motion succeeded.

## Operator Show Integration

Blink now ships a dedicated operator proof-show lane in
`src/blink/embodiment/robot_head/show.py` and `blink-local-robot-head-show`.

- V3: `investor_head_motion_v3`
- V4: `investor_eye_motion_v4`
- V5: `investor_lid_motion_v5`
- V6: `investor_brow_motion_v6`
- V7: `investor_neck_motion_v7`
- V8: `investor_expressive_motion_v8`

The show runner is Blink-owned. It does not import runtime code from an external
robotics codebase. Earlier reference work informed the initial shape
of the V8 expressive ladder and the proof-lane naming.

## Local Runtime Integration

`blink-local-browser` is the browser/WebRTC daily-use brain surface, with
`browser-zh-melo` and `browser-en-kokoro` as equal primary product profiles.
`blink-local-voice` reuses the same brain runtime for backend isolation and
optional operator/debug flows. Both runtimes now layer
robot embodiment on top of Blink's shared identity, memory, and presence stack
instead of a separate local JSON memory shim.

`blink-local-voice` and `blink-local-browser` now accept:

```bash
blink-local-voice --robot-head-driver preview
blink-local-voice --robot-head-driver mock
blink-local-voice --robot-head-driver simulation
blink-local-voice --robot-head-driver live
blink-local-voice --robot-head-catalog-path /path/to/catalog.json
blink-local-voice --robot-head-port /dev/cu.usbmodemXXXX
blink-local-voice --robot-head-live-arm
blink-local-voice --robot-head-hardware-profile-path /path/to/hardware.json
blink-local-voice --robot-head-driver simulation --robot-head-sim-scenario /path/to/scenario.json

blink-local-browser --robot-head-driver preview
blink-local-browser --robot-head-driver simulation
blink-local-browser --robot-head-driver live
blink-local-browser --robot-head-port /dev/cu.usbmodemXXXX
blink-local-browser --robot-head-live-arm --vision
blink-local-browser --robot-head-driver simulation --vision
blink-local-browser --robot-head-driver simulation --vision --no-continuous-perception
blink-local-browser --robot-head-driver simulation --vision --continuous-perception-interval-secs 5
```

`blink-local-robot-head-show` provides the operator proof lane:

```bash
blink-local-robot-head-show --list-shows
blink-local-robot-head-show v3
blink-local-robot-head-show v3 --robot-head-driver simulation --robot-head-sim-scenario /path/to/scenario.json
blink-local-robot-head-show v3 --robot-head-driver preview
blink-local-robot-head-show v3 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v8 --robot-head-driver live --robot-head-live-arm --allow-sensitive-motion
```

Equivalent environment variables:

```bash
BLINK_LOCAL_ROBOT_HEAD_DRIVER=preview
BLINK_LOCAL_ROBOT_HEAD_CATALOG_PATH=/path/to/catalog.json
BLINK_LOCAL_ROBOT_HEAD_PORT=/dev/cu.usbmodemXXXX
BLINK_LOCAL_ROBOT_HEAD_BAUD=1000000
BLINK_LOCAL_ROBOT_HEAD_HARDWARE_PROFILE_PATH=/path/to/hardware.json
BLINK_LOCAL_ROBOT_HEAD_ARM=1
BLINK_LOCAL_ROBOT_HEAD_ARM_TTL_SECONDS=300
BLINK_LOCAL_ROBOT_HEAD_SIM_SCENARIO=/path/to/scenario.json
BLINK_LOCAL_ROBOT_HEAD_SIM_REALTIME=1
BLINK_LOCAL_ROBOT_HEAD_SIM_TRACE_DIR=/path/to/sim-traces
```

When the driver is enabled in either local runtime:

- the default runtime registers only bounded daily-use `robot_head_*` actions
- the LLM can satisfy explicit head-movement requests without raw hardware
  access or raw state names
- the embodiment policy processor runs automatically in the same conversation
  pipeline
- the same browser/voice runtime persists identity blocks, typed memory,
  episodes, tasks, presence snapshots, and action events under
  `~/.cache/blink/brain/brain.db`

When the browser runtime is started with both `--vision` and
`--robot-head-driver simulation`:

- the robot path stays fully hardware-free
- the new symbolic perception broker can keep scene/engagement/body state fresh
  between explicit camera questions
- the detailed camera inspection path still goes only through
  `fetch_user_image`
- the embodiment policy remains bounded; it reacts through the same finite
  robot-head action registry instead of raw hardware control
- any legacy `~/.cache/blink/local_brain/memory.json` facts are treated as
  one-time migration input only
- preview traces are written under `artifacts/robot_head_preview/`
- simulation traces are written under `artifacts/robot_head_simulation/`
- in `live` mode, Blink probes the real bus read-only even when motion is not
  armed, so operator status can confirm port reachability and servo presence
  before motion begins
- the browser/WebRTC path reuses the same controller and policy path instead of
  maintaining a separate robot-control stack

`simulation` is now the preferred hardware-free backend for development and CI.
Unlike `preview`, it preserves deterministic actuator state, timing, telemetry,
and scripted degraded modes while still avoiding all serial and hardware
dependencies.

In the default daily-use browser path, Blink no longer exposes raw
`robot_head_set_state` or `robot_head_run_motif` tools. Those remain available
only behind explicit operator mode for calibration, diagnostics, and proof-show
work. Normal browser conversation goes through the finite embodied action
library first.

## Live Driver Behavior

- The live driver opens the Mac serial port directly and speaks Feetech packets
  from Blink-owned code in `src/blink/embodiment/robot_head/serial_protocol.py`.
- Status probes ping all expected servo IDs, then read positions, voltage,
  temperature, current, and status bytes from the responsive subset.
- Motion compilation stays semantic until the driver boundary. The planner
  never sees servo IDs or raw counts.
- Every live command is compiled against the Blink-owned hardware profile,
  written with preset-specific conservative kinetics, and read back after
  execution.
- Neutral recovery and truly minimal actions stay on `conversation_safe`.
- Public gaze, blink, and social-response motion now uses
  `conversation_readable`, which is slower and materially larger than the
  earlier Blink conversational lane.
- V3-V6 proof motion uses `operator_proof_safe`.
- V7 neck proof uses the slower `operator_neck_safe`.
- V8 expressive proof uses `operator_expressive_v8`, aligned to the safer
  `90/24` expressive lane observed in the maintained reference docs.
- Neck tilt now compiles against the family-specific isolated-tilt targets from
  the checked-in live docs instead of the full raw neck-pair span.
- Startup and shutdown both recover to neutral when Blink has actually moved
  the head in that session.

## Future Work

The remaining live work is no longer “connect the head somehow.” The remaining
work is targeted hardening:

- allow loading a freshly captured Blink-side calibration snapshot instead of
  only the packaged profile mirror
- add an operator-facing doctor / bring-up CLI rather than relying only on the
  voice runtime and proof-show CLI
- capture family-specific live findings back into Blink docs as the head is
  revalidated on this branch
- expand beyond the current staged readable lane only after live proof
  evidence justifies it

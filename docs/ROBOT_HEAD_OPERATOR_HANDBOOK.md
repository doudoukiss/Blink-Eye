# Robot Head Operator Handbook

This handbook is the strict operator reference for commanding Blink's robot
head. It defines what commands are effective now, which control surface they
belong to, which requests are intentionally rejected, and how the larger
expressions decompose into simple hardware-facing semantic movement units.

Use this document when you need exact operational boundaries, not a general
product overview.

## Control Surfaces

There are three different command surfaces. Do not mix them mentally.

### 1. Daily-use browser or voice runtime

This is the normal user-facing path through `blink-local-browser` or
`blink-local-voice` without `--robot-head-operator-mode`.

Allowed surface:

- finite `cmd_*` actions only
- automatic `auto_*` policy actions
- no raw state names
- no raw motif names
- no neck commands
- no multi-action free composition

### 2. Operator mode in browser or voice runtime

This is the same runtime started with `--robot-head-operator-mode`.

Allowed surface:

- exact persistent states
- exact public motifs
- neutral return
- status

Use this only for calibration, diagnostics, and controlled operator demos.

### 3. Operator proof-show CLI

This is the standalone proof lane:

```bash
blink-local-robot-head-show ...
```

Allowed surface:

- V3 through V8 proof shows
- sensitive neck and expressive lanes when explicitly armed and allowed

Use this when you want deterministic family proof or expressive proof motion,
not conversational control.

## Hard Rules

- Daily-use browser commands must be single-intent. One utterance should request one physical action.
- Do not ask for chained movement such as `先眨眼再看左边再回中位`.
- Do not ask for raw units or actuator semantics such as `servo 1`, `舵机`, `串口`, `head_turn`, or `eye_yaw`.
- Do not ask the daily path for neck pitch or neck tilt.
- Use `回到中位` instead of trying to invent a custom neutral pose.
- Use `现在头部状态是什么？` when you need availability, arm state, mode, or warnings.
- Use the proof-show CLI for V3-V8 family demos. Do not try to trigger those motifs from the normal browser path.

## Daily-use Runtime Commands

These are the commands that are effective in the normal browser path today.
They map to the finite embodied action library in
`src/blink/brain/actions.py`.

| Recommended operator phrase | Action id | Resolved controller plan | Semantic units touched |
| --- | --- | --- | --- |
| `眨眼一次` | `cmd_blink` | `run_motif(blink)` | `left_lids`, `right_lids` |
| `左眼眨一下` | `cmd_wink_left` | `run_motif(wink_left)` | `left_lids` |
| `右眼眨一下` | `cmd_wink_right` | `run_motif(wink_right)` | `right_lids` |
| `向左看一下` | `cmd_look_left` | `run_motif(look_left)` | `head_turn`, `eye_yaw` |
| `向右看一下` | `cmd_look_right` | `run_motif(look_right)` | `head_turn`, `eye_yaw` |
| `回到中位` | `cmd_return_neutral` | `return_neutral()` | neutral recovery |
| `现在头部状态是什么？` | `cmd_report_status` | `status()` | no motion |

Recommended English phrases:

- `blink`
- `wink left`
- `wink right`
- `look left`
- `look right`
- `return to neutral`
- `head status`

## Automatic Conversational Actions

These are not operator utterances. Blink triggers them automatically through
the embodiment policy.

| Trigger | Action id | Resolved controller plan |
| --- | --- | --- |
| user starts speaking | `auto_listen_user` | `run_motif(listen_engage)` -> `set_state(listen_attentively)` |
| user stops speaking | `auto_think` | `run_motif(thinking_shift)` -> `set_state(thinking)` |
| Blink starts speaking | `auto_speak_friendly` | `run_motif(acknowledge)` -> `set_state(friendly)` |
| idle, cancel, stop, error | `auto_safe_idle` | `set_state(safe_idle)` -> `return_neutral()` |

This is why the robot reacts while people speak. That behavior is intentional.
It is the policy layer, not uncontrolled servo chatter.

## Operator-Mode Browser Or Voice Commands

Start operator mode only when you need exact state or motif control:

```bash
./scripts/run-blink-browser.sh \
  --robot-head-driver simulation \
  --robot-head-sim-scenario /path/to/scenario.json \
  --robot-head-operator-mode

./scripts/run-blink-browser.sh \
  --robot-head-driver live \
  --robot-head-port /dev/cu.usbmodemXXXX \
  --robot-head-live-arm \
  --robot-head-operator-mode
```

Or:

```bash
./scripts/run-blink-voice.sh \
  --robot-head-driver simulation \
  --robot-head-sim-scenario /path/to/scenario.json \
  --robot-head-operator-mode

./scripts/run-blink-voice.sh \
  --robot-head-driver live \
  --robot-head-port /dev/cu.usbmodemXXXX \
  --robot-head-live-arm \
  --robot-head-operator-mode
```

In operator mode, prefer exact state or motif names in the request.

### Exact public states

| Recommended operator phrase | Tool intent | Units touched |
| --- | --- | --- |
| `切换到 friendly 状态` | `robot_head_set_state(friendly)` | lids, brows, eye pitch |
| `切换到 listen_attentively 状态` | `robot_head_set_state(listen_attentively)` | lids, brows, eye pitch |
| `切换到 thinking 状态` | `robot_head_set_state(thinking)` | lids, brows, eye pitch |
| `切换到 focused_soft 状态` | `robot_head_set_state(focused_soft)` | lids, brows, eye pitch |
| `切换到 confused 状态` | `robot_head_set_state(confused)` | lids, brows, eye pitch |
| `切换到 safe_idle 状态` | `robot_head_set_state(safe_idle)` | lids |
| `回到中位` | `robot_head_return_neutral()` | neutral recovery |

`neutral` exists in the catalog, but operators should still prefer
`回到中位` so the controller uses the neutral-return path directly.

### Exact public motifs

| Recommended operator phrase | Tool intent | Units touched |
| --- | --- | --- |
| `执行 acknowledge 动作` | `robot_head_run_motif(acknowledge)` | `head_turn`, `eye_yaw`, lids, brows |
| `执行 blink 动作` | `robot_head_run_motif(blink)` | `left_lids`, `right_lids` |
| `执行 wink_left 动作` | `robot_head_run_motif(wink_left)` | `left_lids` |
| `执行 wink_right 动作` | `robot_head_run_motif(wink_right)` | `right_lids` |
| `执行 look_left 动作` | `robot_head_run_motif(look_left)` | `head_turn`, `eye_yaw` |
| `执行 look_right 动作` | `robot_head_run_motif(look_right)` | `head_turn`, `eye_yaw` |
| `现在头部状态是什么？` | `robot_head_status()` | no motion |

Do not rely on operator mode for V3-V8 proof motion. Use the show CLI instead.

## Proof-Show CLI Commands

These are the exact deterministic operator proof commands.

```bash
blink-local-robot-head-show --list-shows
blink-local-robot-head-show v3
blink-local-robot-head-show v3 --robot-head-driver simulation --robot-head-sim-scenario /path/to/scenario.json
blink-local-robot-head-show v3 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v4 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v5 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v6 --robot-head-driver live --robot-head-live-arm
blink-local-robot-head-show v7 --robot-head-driver live --robot-head-live-arm --allow-sensitive-motion
blink-local-robot-head-show v8 --robot-head-driver live --robot-head-live-arm --allow-sensitive-motion
```

Proof-show coverage:

- `v3`: head yaw only
- `v4`: eye yaw and eye pitch
- `v5`: lid motion and blink
- `v6`: brow motion
- `v7`: neck tilt and neck pitch
- `v8`: twelve staged composed expressions

## Simple Movement Families

These are the semantic movement units Blink can drive now.

| Unit | Meaning | Daily-use path | Operator proof lane |
| --- | --- | --- | --- |
| `head_turn` | head yaw left or right | yes | yes |
| `eye_yaw` | eyes left or right | yes | yes |
| `eye_pitch` | eyes up or down | automatic state use only | yes |
| `left_lids` | left eyelid close or open | yes | yes |
| `right_lids` | right eyelid close or open | yes | yes |
| `left_brow` | left brow raise or lower | automatic state use only | yes |
| `right_brow` | right brow raise or lower | automatic state use only | yes |
| `neck_pitch` | neck up or down | no | yes |
| `neck_tilt` | neck tilt left or right | no | yes |

The daily-use path is intentionally narrower than hardware capability.

## Public Motion Definitions

These are the exact public motions behind the daily path.

### Blink

- `blink`
  - close: `left_lids=-0.72`, `right_lids=-0.72`
  - open: `left_lids=0.14`, `right_lids=0.14`

### Wink left

- `wink_left`
  - close: `left_lids=-0.72`
  - open: `left_lids=0.14`

### Wink right

- `wink_right`
  - close: `right_lids=-0.72`
  - open: `right_lids=0.14`

### Look left

- `look_left`
  - turn: `head_turn=-0.6`, `eye_yaw=-0.68`
  - return: neutral release

### Look right

- `look_right`
  - turn: `head_turn=0.6`, `eye_yaw=0.68`
  - return: neutral release

### Acknowledge

- `acknowledge`
  - engage: `head_turn=0.24`, `eye_yaw=0.18`, `left_lids=0.16`, `right_lids=0.16`, `left_brow=0.18`, `right_brow=0.18`
  - release: `head_turn=0.1`, `eye_yaw=0.08`, `left_lids=0.14`, `right_lids=0.14`
  - return: neutral release

## Held Public States

These are eye-area held states, not structural poses.

| State | Exact semantic values |
| --- | --- |
| `friendly` | `left_lids=0.24`, `right_lids=0.24`, `left_brow=0.24`, `right_brow=0.24`, `eye_pitch=0.04` |
| `listen_attentively` | `left_lids=0.28`, `right_lids=0.28`, `left_brow=0.26`, `right_brow=0.26`, `eye_pitch=0.12` |
| `thinking` | `left_lids=0.06`, `right_lids=0.06`, `left_brow=0.06`, `right_brow=0.2`, `eye_pitch=-0.14` |
| `focused_soft` | `left_lids=0.16`, `right_lids=0.16`, `left_brow=0.12`, `right_brow=0.12`, `eye_pitch=0.04` |
| `confused` | `left_lids=0.08`, `right_lids=0.04`, `left_brow=0.24`, `right_brow=-0.18`, `eye_pitch=-0.12` |
| `safe_idle` | `left_lids=0.1`, `right_lids=0.1` |

## V3-V8 Decomposition

This section decomposes the larger operator expressions into simple movement
families.

### V3 head motion

- `investor_head_turn_left_v3`
  - `head_turn=-0.82`
- `investor_head_turn_right_v3`
  - `head_turn=0.82`
- `investor_head_sweep_left_v3`
  - prep: `head_turn=0.42`
  - sweep: `head_turn=-0.82`
- `investor_head_sweep_right_v3`
  - prep: `head_turn=-0.42`
  - sweep: `head_turn=0.86`

### V4 eye motion

- `investor_eye_yaw_left_v4`
  - `eye_yaw=-0.82`
- `investor_eye_yaw_right_v4`
  - `eye_yaw=0.82`
- `investor_eye_pitch_up_v4`
  - `eye_pitch=0.62`, `left_lids=0.14`, `right_lids=0.14`
- `investor_eye_pitch_down_v4`
  - `eye_pitch=-0.58`

### V5 lid motion

- `investor_both_lids_v5`
  - close: `left_lids=-0.78`, `right_lids=-0.78`
  - open: `left_lids=0.18`, `right_lids=0.18`
- `investor_left_eye_lids_v5`
  - close: `left_lids=-0.78`
  - open: `left_lids=0.18`
- `investor_right_eye_lids_v5`
  - close: `right_lids=-0.78`
  - open: `right_lids=0.18`
- `investor_blink_v5`
  - close: `left_lids=-0.78`, `right_lids=-0.78`
  - open: `left_lids=0.16`, `right_lids=0.16`

### V6 brow motion

- `investor_brows_both_v6`
  - `left_brow=0.48`, `right_brow=0.48`
- `investor_brow_left_v6`
  - `left_brow=0.52`
- `investor_brow_right_v6`
  - `right_brow=0.52`

### V7 neck motion

- `investor_neck_tilt_left_v7`
  - `neck_tilt=-0.78`, `eye_yaw=-0.18`
- `investor_neck_tilt_right_v7`
  - `neck_tilt=0.78`, `eye_yaw=0.18`
- `investor_neck_pitch_up_v7`
  - `neck_pitch=0.42`, `eye_pitch=0.18`
- `investor_neck_pitch_down_v7`
  - `neck_pitch=-0.42`, `eye_pitch=-0.22`

### V8 composed expressions

- `attentive_notice_right`
  - structural set: `head_turn=0.72`
  - expressive set: `eye_yaw=0.58`, `left_brow=0.36`, `right_brow=0.36`, `left_lids=0.16`, `right_lids=0.16`
  - release: `eye_yaw=0.24`, `left_lids=0.12`, `right_lids=0.12`

- `attentive_notice_left`
  - structural set: `head_turn=-0.72`
  - expressive set: `eye_yaw=-0.58`, `left_brow=0.36`, `right_brow=0.36`, `left_lids=0.16`, `right_lids=0.16`
  - release: `eye_yaw=-0.24`, `left_lids=0.12`, `right_lids=0.12`

- `guarded_close_right`
  - structural set: `head_turn=0.74`
  - expressive set: `eye_yaw=0.56`, `left_lids=-0.78`, `right_lids=-0.78`
  - brow compression: `left_brow=-0.22`, `right_brow=-0.22`
  - release: `eye_yaw=0.32`, `left_lids=0.14`, `right_lids=0.14`

- `guarded_close_left`
  - structural set: `head_turn=-0.74`
  - expressive set: `eye_yaw=-0.56`, `left_lids=-0.78`, `right_lids=-0.78`
  - brow compression: `left_brow=-0.22`, `right_brow=-0.22`
  - release: `eye_yaw=-0.32`, `left_lids=0.14`, `right_lids=0.14`

- `curious_lift`
  - structural set: `neck_pitch=0.34`
  - expressive set: `eye_pitch=0.32`, `left_brow=0.42`, `right_brow=0.42`, `left_lids=0.18`, `right_lids=0.18`
  - release: `neck_pitch=0.14`, `eye_pitch=0.14`, `left_brow=0.18`, `right_brow=0.18`

- `reflective_lower`
  - structural set: `neck_pitch=-0.34`
  - expressive set: `eye_pitch=-0.3`, `left_brow=-0.18`, `right_brow=-0.18`, `left_lids=0.08`, `right_lids=0.08`
  - release: `neck_pitch=-0.14`, `eye_pitch=-0.14`, `left_brow=-0.08`, `right_brow=-0.08`

- `skeptical_tilt_right`
  - structural set: `neck_tilt=0.72`
  - expressive set: `eye_yaw=0.34`, `left_brow=0.38`, `right_brow=-0.2`, `left_lids=0.12`, `right_lids=0.04`
  - release: `neck_tilt=0.32`, `eye_yaw=0.12`

- `empathetic_tilt_left`
  - structural set: `neck_tilt=-0.72`
  - expressive set: `left_lids=0.18`, `right_lids=0.18`, `left_brow=0.24`, `right_brow=0.24`, `eye_pitch=-0.06`
  - release: `neck_tilt=-0.32`, `left_lids=0.12`, `right_lids=0.12`, `left_brow=0.12`, `right_brow=0.12`

- `playful_peek_right`
  - structural set: `head_turn=0.5`
  - expressive set: `eye_yaw=0.48`, `right_lids=-0.78`, `left_brow=0.22`, `right_brow=0.4`
  - release: `head_turn=0.32`, `eye_yaw=0.22`, `right_lids=0.14`

- `playful_peek_left`
  - structural set: `head_turn=-0.5`
  - expressive set: `eye_yaw=-0.48`, `left_lids=-0.78`, `left_brow=0.4`, `right_brow=0.22`
  - release: `head_turn=-0.32`, `eye_yaw=-0.22`, `left_lids=0.14`

- `bright_reengage`
  - structural set: `head_turn=-0.46`
  - expressive set: `eye_yaw=-0.42`, `eye_pitch=0.12`, `left_lids=0.24`, `right_lids=0.24`, `left_brow=0.42`, `right_brow=0.42`
  - release: `head_turn=-0.24`, `eye_yaw=-0.16`, `left_lids=0.14`, `right_lids=0.14`, `left_brow=0.18`, `right_brow=0.18`

- `doubtful_side_glance`
  - structural set: `head_turn=-0.44`
  - expressive set: `eye_yaw=-0.52`, `left_brow=0.12`, `right_brow=0.42`, `left_lids=0.08`, `right_lids=0.02`
  - release: `head_turn=-0.22`, `eye_yaw=-0.18`

## Requests That Should Be Rejected

These requests are outside the supported command surface and should be treated
as unsupported:

- `把 1 号舵机转到 2300`
- `把 eye_yaw 调到 0.7`
- `做一个新的表情`
- `先看左边再眨眼再抬头`
- `向左歪头`
- `低头`
- `连续表演所有表情`
- `按你觉得自然的方式自己动`

Use one of the supported daily commands, operator-mode state or motif names, or
the proof-show CLI instead.

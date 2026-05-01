# Autonomous Presence Director QA

Use this runbook after landing any Phase 6 Presence Director work.

The objective is to verify that Blink becomes more intentional and legible
without becoming noisy, unsafe, or overly eager.

## Setup

Recommended path:

```bash
mkdir -p artifacts/manual_test
export BLINK_LOCAL_BRAIN_DB_PATH="$PWD/artifacts/manual_test/brain.db"
./scripts/run-local-browser-melo.sh
```

When testing maintenance or audit behavior, also keep these commands handy:

```bash
uv run --python 3.12 blink-local-brain-audit --runtime-kind browser
uv run --python 3.12 blink-local-brain-reflect --runtime-kind browser
```

## Fast 3-Minute Manual Validation

Use this when you want one short human-in-front-of-camera check instead of the
full scenario set.

### Start a clean browser run

In Terminal A:

```bash
export BLINK_LOCAL_BRAIN_DB_PATH=/tmp/blink_presence_manual.db
rm -f "$BLINK_LOCAL_BRAIN_DB_PATH"
./scripts/run-local-browser-melo.sh --port 7864
```

In the browser:

```text
http://127.0.0.1:7864/client/
```

Allow camera and microphone, then click `Connect`.

Expected:

- the page reaches `Client READY`
- the page reaches `Agent READY`

### Minute 1: re-entry and attention return

1. Stay fully out of frame for about 5 seconds.
2. Step into frame and look directly at the camera for 5 to 8 seconds.
3. While still in frame, look away for 5 seconds, then look back at the camera.

Expected:

- the re-entry path may create `presence_user_reentered`
- the attention path may create `presence_attention_returned`
- if Blink does not act, the audit trail should still show `suppressed` or
  `non_action` instead of silent disappearance

### Minute 2: spoken initiative gate

1. Leave the frame for about 12 seconds.
2. Re-enter the frame, face the camera, and stay silent.
3. Repeat once more, but start speaking immediately after re-entering.

Expected:

- the silent return may lead to the narrow
  `presence_brief_reengagement_speech` path
- the speaking case should not talk over the user
- if Blink holds the initiative, the reason should be visible as
  `user_turn_open`, `assistant_turn_open`, or another stable policy code

### Minute 3: stop and inspect

Stop Terminal A with `Ctrl+C`, then run:

```bash
uv run --python 3.12 blink-local-brain-audit \
  --runtime-kind browser \
  --brain-db-path "$BLINK_LOCAL_BRAIN_DB_PATH"
```

Inspect the candidate and decision trail:

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "
select
  event_type,
  ts,
  json_extract(payload_json, '$.candidate_goal.candidate_type'),
  json_extract(payload_json, '$.candidate_goal.summary'),
  json_extract(payload_json, '$.candidate_goal_id'),
  json_extract(payload_json, '$.reason')
from brain_events
where event_type in (
  'goal.candidate.created',
  'goal.candidate.suppressed',
  'goal.candidate.merged',
  'goal.candidate.accepted',
  'director.non_action.recorded'
)
order by ts desc
limit 30;
"
```

Inspect the perception side too:

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "
select
  event_type,
  ts,
  json_extract(payload_json, '$.summary'),
  json_extract(payload_json, '$.camera_fresh'),
  json_extract(payload_json, '$.person_present'),
  json_extract(payload_json, '$.confidence')
from brain_events
where event_type in ('scene.changed', 'perception.observed')
order by ts desc
limit 30;
"
```

Pass criteria:

- the browser session reaches `READY`
- the audit contains both `autonomy_ledger` and `autonomy_digest`
- at least one re-entry or attention-return step leaves a visible candidate
  decision trail when camera health is good
- if Blink does nothing, `visual_health` explains why

Interpretation:

- no autonomy events plus `healthy` camera state is a real regression
- no autonomy events plus `recovering`, `stalled`, or `uncertain` visual state
  usually means the perception input never became strong enough

## Scenario 1: person re-enters frame while Blink is idle

Steps:

1. Start with the camera on and Blink already stable.
2. Step out of frame long enough for absence to register.
3. Step back into frame and wait a few seconds.

Expected:

- a candidate goal is created from the scene transition
- Blink either performs one tasteful silent attention shift or records a
  suppression / non-action decision
- no repeated motion loop occurs from the same transition

## Scenario 2: user is speaking when a perception candidate becomes eligible

Steps:

1. Start talking continuously while re-entering frame or restoring strong
   attention to camera.
2. Let Blink observe the state change during the user turn.

Expected:

- the candidate is not allowed to interrupt with unsolicited speech
- the ledger records `non_action` with `user_turn_open`
- the candidate may remain available for reevaluation after the turn ends

## Scenario 3: degraded or flapping camera input

Steps:

1. Temporarily stall the camera, partially occlude it, or create a noisy
   near-threshold situation.
2. Let the perception broker emit several observations.

Expected:

- Blink becomes more conservative, not more animated
- low-confidence or duplicate signals are recorded as `suppressed` or `merged`
- common reasons now include `low_confidence`, `cooldown_active`,
  `duplicate_active_goal`, and `goal_family_busy`
- no repetitive oscillation appears in action or non-action trails

## Scenario 4: deferred commitment becomes eligible

Steps:

1. Create a small deferred commitment in a browser or text session.
2. Make its wake condition eligible.
3. Allow a startup pass, turn-end pass, or timer-driven reevaluation.

Expected:

- a candidate goal or equivalent director input is created
- Blink either accepts it into agenda work or records a clean `non_action`
  or `suppressed` reason
- unsolicited long speech still stays gated by policy

## Scenario 5: maintenance interval elapses during idle time

Steps:

1. Leave Blink idle long enough for a maintenance wake.
2. Trigger or wait for the maintenance scheduler or director timer.

Expected:

- bounded maintenance work runs quietly, or a defer/non-action reason is
  recorded if the timing is wrong
- maintenance does not preempt active user interaction
- audit or review artifacts remain inspectable afterwards

## Scenario 6: replay and audit explain behavior

Steps:

1. Run a short session containing one accepted candidate and one suppressed
   or non-action candidate.
2. Generate the continuity audit and inspect replay/export output.

Expected:

- the autonomy ledger shows the candidate lifecycle clearly
- operator-visible artifacts explain why Blink acted or did not act
- accepted, suppressed, merged, and non-action cases are distinguishable

## Scenario 7: operator reviews a no-action case after the fact

Steps:

1. Trigger a case that should be held, for example spoken re-engagement while
   the user is still speaking.
2. Run `uv run --python 3.12 blink-local-brain-audit --runtime-kind browser`.
3. Open the generated JSON and Markdown artifacts.

Expected:

- `autonomy_digest` summarizes the case without reading raw lifecycle rows
- the raw `autonomy_ledger` still shows the exact non-action event
- the reason is one of the stable policy codes such as `user_turn_open`,
  `assistant_turn_open`, or `goal_family_busy`
- the reevaluation condition is present when the non-action is expected to be
  revisited later

## Automatic Coverage

Use the automated lane to prove the bounded policy and replay surfaces before
running the manual camera check.

Core Phase 6 coverage:

```bash
uv run pytest \
  tests/test_brain_perception_broker.py \
  tests/test_brain_commitments.py \
  tests/test_brain_autonomy.py \
  tests/test_brain_capability_registry.py \
  tests/test_brain_runtime.py \
  tests/test_brain_replay.py \
  tests/test_brain_audit_reports.py -q
```

This covers:

- perception-driven candidate production
- commitment-wake and maintenance candidates
- Presence Director suppress / merge / accept / non-action policy
- capability-family and initiative-gate boundaries
- replay and audit visibility for action and non-action cases

Optional browser UI smoke test:

```bash
uv run python -m playwright install chromium
BLINK_RUN_BROWSER_E2E=1 uv run pytest tests/test_browser_e2e.py -q
```

That browser smoke proves:

- the browser UI loads
- the client can connect to the local SmallWebRTC runtime
- fake camera/microphone permissions are sufficient for a connection

It does not replace the manual three-minute camera check above. The browser E2E
path uses fake media and may still leave `person_present` as `uncertain`, so
the final re-entry and spoken-initiative validation still needs a real human
camera session.

## Score dimensions

Rate each from 1 to 5:

- initiative tastefulness
- suppression quality
- non-action explainability
- degraded-mode safety
- maintenance discipline
- operator trust

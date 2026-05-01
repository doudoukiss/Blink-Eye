# Autonomy Reevaluation QA

Use this runbook after landing any Phase 7 autonomy reevaluation work.

## Goal

Verify that current candidates can be held, reconsidered, accepted, expired,
and audited **without requiring a fresh duplicate proposal**.

## Core scenarios

### Scenario 1: held candidate resumes after user turn closes

1. Produce a scene or dialogue candidate that requires a turn gap.
2. Keep the user turn open so Blink records non-action.
3. End the user turn.
4. Confirm Blink reevaluates the same current candidate instead of waiting for a
   brand-new duplicate producer event.

Expected result:

- a non-action record exists first
- the later reevaluation is visible in events or derived audit output
- the candidate is eventually accepted, suppressed, or expired
- the candidate does not remain current forever
- the same `candidate_goal_id` continues the lifecycle without a fresh duplicate
  `goal.candidate.created`

### Scenario 2: idle-window maintenance candidate eventually runs

1. Trigger a maintenance candidate during active thread use.
2. Confirm Blink records a hold / non-action reason.
3. Leave the thread idle long enough for the reevaluation window.
4. Confirm bounded maintenance work runs or the candidate expires clearly.

### Scenario 3: expired current candidate is cleaned up without unrelated traffic

1. Produce a short-lived candidate with an expiry window.
2. Do not create any unrelated new candidate.
3. Wait for the expiry trigger or run the dedicated reevaluation lane.

Expected result:

- the candidate disappears from `current_candidates`
- an explicit expiry trail exists
- cleanup does not require an unrelated fresh proposal

### Scenario 4: startup recovery reevaluates pending candidates

1. Leave one candidate pending in the ledger.
2. Restart the runtime.
3. Confirm startup reevaluation considers that pending candidate.

### Scenario 5: fairness across candidate families

1. Produce repeated scene candidates.
2. Produce a maintenance or commitment-wake candidate.
3. Confirm one family does not starve the other indefinitely.

## Direct SQLite inspection

Use the local brain DB and inspect:

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_type, ts, json_extract(payload_json, '$.candidate_goal_id'), json_extract(payload_json, '$.reason') from brain_events where event_type in ('goal.candidate.created', 'goal.candidate.suppressed', 'goal.candidate.merged', 'goal.candidate.accepted', 'goal.candidate.expired', 'director.non_action.recorded') order by ts desc limit 40;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_type, ts, json_extract(payload_json, '$.trigger.kind'), json_extract(payload_json, '$.trigger.summary'), json_extract(payload_json, '$.candidate_goal_ids') from brain_events where event_type = 'director.reevaluation.triggered' order by ts desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select projection_json from brain_projections where projection_name='autonomy_ledger' order by updated_at desc limit 1;"
```

## Audit review

In the continuity audit JSON or Markdown, inspect:

- `autonomy_ledger` for the raw candidate lifecycle
- `autonomy_digest` for the compact current / accepted / suppressed / merged /
  non-action operator summary
- `reevaluation_digest` for current holds, trigger kinds, and recent
  hold -> reevaluation -> outcome flows

For the explicit operator question "why did Blink do nothing, and what happened
later?", confirm you can answer all three:

- which candidate was held and why
- which reevaluation trigger later fired
- whether the result was accept, non-action again, suppression, or expiry

## What a good run proves

- non-action is not a dead end
- reevaluation conditions are actually owned by the runtime
- expiry cleanup is real
- pending candidates remain legible to operators
- the next decision is driven by the same candidate lifecycle, not by a lucky
  duplicate event

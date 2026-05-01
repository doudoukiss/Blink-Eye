# Commitment Wake Router QA

Use this runbook after changing any Phase 8 wake-router behavior.

## Goal

Verify that deferred and blocked commitments remain inspectable while the
runtime routes them through one bounded wake outcome:

- `resume_direct`
- `propose_candidate`
- `keep_waiting`

## Primary Operator Surfaces

Check these in order:

1. `wake_digest` in the continuity audit or thread digest
2. `commitment_projection` for the current waiting commitments
3. raw `commitment.wake.triggered` events in SQLite
4. the downstream causal chain to `goal.resumed` or `goal.candidate.created`

`operator_review` is still inspectable wake metadata only in this phase. It is
not a runtime-owned automatic wake.

## Scenario 1: thread-idle wake proposes bounded candidate work

1. Create a commitment.
2. Defer it with a `thread_idle` wake condition.
3. Let the thread become idle.
4. Confirm one `commitment.wake.triggered` event is recorded.
5. Confirm the route is `propose_candidate`.
6. Confirm one downstream `goal.candidate.created` points back to that wake by
   `causal_parent_id`.

Expected:

- `wake_digest.recent_candidate_proposals` contains the commitment title,
  wake kind, candidate id, and candidate type
- `wake_digest.route_counts.propose_candidate >= 1`
- the same commitment remains visible in `wake_digest.current_waiting_commitments`

## Scenario 2: user-response wake still requires a newer user event

1. Create a deferred commitment with `user_response`.
2. Run the router without a newer user-turn event.
3. Confirm no wake is routed.
4. Record a newer `user.turn.ended` or equivalent user-turn event.
5. Confirm the next bounded router pass emits one wake trigger.

Expected:

- no duplicate wake traffic for the same source event
- the routed wake remains legible through `wake_digest.recent_triggers`

## Scenario 3: condition-cleared wake resumes directly

1. Block a commitment on a machine-checkable capability blocker.
2. Attach a `condition_cleared` wake condition with a concrete
   `capability_id`.
3. Clear the blocker and run one bounded router boundary.
4. Confirm the runtime emits `commitment.wake.triggered`.
5. Confirm the route is `resume_direct`.
6. Confirm one downstream `goal.resumed` points back to that wake by
   `causal_parent_id`.

Expected:

- `wake_digest.recent_direct_resumes` contains the commitment title and resumed
  goal linkage
- `wake_digest.route_counts.resume_direct >= 1`
- the resumed commitment is no longer listed in
  `wake_digest.current_waiting_commitments`

## Scenario 4: why Blink kept waiting

1. Create a deferred or blocked commitment whose wake would otherwise match.
2. Keep an inspectable blocker in place, such as an already-current wake
   candidate.
3. Run one bounded router boundary.
4. Confirm one `commitment.wake.triggered` event is recorded with
   `route_kind = keep_waiting`.
5. Confirm there is no downstream `goal.resumed` or `goal.candidate.created`
   for that wake event id.

Expected:

- `wake_digest.recent_keep_waiting` contains the commitment title, wake kind,
  `reason`, and `boundary_kind`
- `wake_digest.reason_counts` includes the stable reason code, for example
  `candidate_already_current`
- operators can answer "why Blink kept waiting" from audit JSON/Markdown plus
  SQLite alone

## Scenario 5: explicit-resume remains manual

1. Create a blocked commitment with `explicit_resume`.
2. Run startup, turn-end, user-turn-close, and goal-terminal automatic
   boundaries.
3. Confirm no automatic wake trigger or resume occurs.
4. Resume it explicitly.
5. Confirm the commitment resumes only then.

Expected:

- no automatic `commitment.wake.triggered` event for the `explicit_resume`
  case
- the commitment remains inspectable in the commitment projection

## Scenario 6: restart recovery preserves wake state

1. Leave at least one commitment waiting on a runtime-owned wake condition.
2. Restart the runtime.
3. Confirm the commitment is still present in `commitment_projection`.
4. Confirm later matching boundary changes still produce the same wake kind and
   route kind deterministically.

## Useful Inspection Commands

Use the DB from `BLINK_LOCAL_BRAIN_DB_PATH` if you overrode it.

### Raw wake events

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "
select
  event_id,
  event_type,
  ts,
  json_extract(payload_json, '$.commitment.commitment_id') as commitment_id,
  json_extract(payload_json, '$.trigger.wake_kind') as wake_kind,
  json_extract(payload_json, '$.routing.route_kind') as route_kind,
  json_extract(payload_json, '$.routing.details.reason') as reason
from brain_events
where event_type = 'commitment.wake.triggered'
order by ts desc
limit 40;
"
```

### Wake -> direct resume causal chain

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "
select
  wake.event_id as wake_event_id,
  wake.ts as wake_ts,
  json_extract(wake.payload_json, '$.commitment.commitment_id') as commitment_id,
  json_extract(wake.payload_json, '$.trigger.wake_kind') as wake_kind,
  json_extract(wake.payload_json, '$.routing.route_kind') as route_kind,
  resumed.event_id as resumed_event_id,
  json_extract(resumed.payload_json, '$.goal.goal_id') as resumed_goal_id,
  json_extract(resumed.payload_json, '$.goal.title') as resumed_goal_title
from brain_events wake
left join brain_events resumed
  on resumed.causal_parent_id = wake.event_id
 and resumed.event_type = 'goal.resumed'
where wake.event_type = 'commitment.wake.triggered'
order by wake.ts desc
limit 20;
"
```

### Wake -> candidate proposal causal chain

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "
select
  wake.event_id as wake_event_id,
  wake.ts as wake_ts,
  json_extract(wake.payload_json, '$.commitment.commitment_id') as commitment_id,
  json_extract(wake.payload_json, '$.trigger.wake_kind') as wake_kind,
  json_extract(wake.payload_json, '$.routing.route_kind') as route_kind,
  candidate.event_id as candidate_event_id,
  json_extract(candidate.payload_json, '$.candidate_goal.candidate_goal_id') as candidate_goal_id,
  json_extract(candidate.payload_json, '$.candidate_goal.candidate_type') as candidate_type
from brain_events wake
left join brain_events candidate
  on candidate.causal_parent_id = wake.event_id
 and candidate.event_type = 'goal.candidate.created'
where wake.event_type = 'commitment.wake.triggered'
order by wake.ts desc
limit 20;
"
```

## What to Verify in Artifacts

- `wake_digest.current_waiting_commitments` shows the waiting commitment title,
  goal family, wake kinds, plan revision, and resume count
- `wake_digest.trigger_counts`, `route_counts`, and `reason_counts` match the
  raw wake events
- `wake_digest.recent_direct_resumes`,
  `wake_digest.recent_candidate_proposals`, and
  `wake_digest.recent_keep_waiting` make action and non-action equally legible
- the raw wake event appears before the downstream `goal.resumed` or
  `goal.candidate.created`
- `keep_waiting` cases show no fake downstream progress
- `explicit_resume` cases stay dormant unless resumed manually

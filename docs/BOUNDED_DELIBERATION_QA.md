# Bounded Deliberation and Plan Revision QA

Use this runbook after changing any Phase 9 planning / revision behavior.

The phase is about making plan proposal, adoption, rejection, and revision as
inspectable as wake routing already is.

## Automatic coverage

Run at least:

```bash
uv run pytest \
  tests/test_brain_planning.py \
  tests/test_brain_commitments.py \
  tests/test_brain_context_policy.py \
  tests/test_brain_replay.py \
  tests/test_brain_audit_reports.py \
  tests/test_brain_layered_memory.py \
  tests/test_brain_import_hygiene.py -q
```

What this automatic lane should prove after the phase lands:

- planning-needed work can emit typed plan proposals
- adopted revisions preserve completed prefixes
- replay keeps the same current `plan_revision` and proposal history
- `planning_digest` stays deterministic across repeated replays
- audit JSON / Markdown expose plan history as clearly as wake history
- lower planning/executive collection stays provider-light enough for focused testing

## Operator-first review order

For any Phase 9 planning or revision run, inspect in this order:

1. `planning_digest` in the continuity audit or thread digest
2. current goal / commitment `details` fields such as `current_plan_proposal_id`, `pending_plan_proposal_id`, and `plan_review_policy`
3. raw `planning.requested`, `planning.proposed`, `planning.adopted`, and `planning.rejected` events
4. downstream `goal.updated` or `goal.repaired` events for the same causal chain

The operator source of truth is raw planning events plus the derived `planning_digest`, not screenshots.

## Manual scenario 1: planning-required work is no longer a dead end

1. Create or replay a goal that has no explicit capability tail and no existing deterministic autonomy template.
2. Run one bounded executive cycle.
3. Inspect recent events and commitment state.

Expected outcome:

- you still see `planning.requested`
- you also see `planning_digest`
- you also see an inspectable `planning.proposed` event or equally explicit deferred-planning state
- the commitment does not disappear into an opaque blocked state with no further plan trail

## Manual scenario 2: revision preserves completed prefix

1. Start with a multi-step commitment.
2. Complete at least one step.
3. Trigger a revision path for the remaining tail.
4. Inspect the adopted revision.

Expected outcome:

- completed steps remain unchanged
- only the remaining tail changes
- `plan_revision` increments
- `planning_digest.recent_revision_flows` shows the preserved prefix and downstream `goal.repaired` linkage
- earlier revision history stays inspectable

## Manual scenario 3: review boundary is respected

1. Trigger a plan proposal that changes dialogue or embodiment-facing behavior.
2. Inspect the proposal policy flags and resulting state.

Expected outcome:

- risky revisions are not silently auto-adopted
- the system records whether user or operator review is required
- the waiting reason is visible in `planning_digest`, current goal / commitment details, and raw planning events

## Manual scenario 4: wake and planning compose correctly

1. Create a deferred or blocked commitment with a runtime-ownable wake condition.
2. Trigger the wake.
3. Ensure the resumed work now requires a revised tail.
4. Inspect recent wake and planning events together.

Expected outcome:

- wake ownership remains explicit
- the system either resumes against the latest valid plan revision or requests a new plan revision explicitly
- operators can follow the trail from `commitment.wake.triggered` to the plan outcome in raw events and `planning_digest`

## Manual scenario 5: replay preserves current revision

1. Produce at least one proposal and one adopted revision.
2. Export or replay the relevant event stream.
3. Rebuild projections from replay.

Expected outcome:

- the same current revision remains current
- the same proposal / adoption history remains visible
- the same `planning_digest` remains stable across repeated replays
- replay does not silently mutate the adopted tail

## Direct SQLite inspection

Use these queries alongside the audit digest:

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_id, event_type, ts, json_extract(payload_json, '$.goal_id'), json_extract(payload_json, '$.commitment_id'), json_extract(payload_json, '$.proposal.plan_proposal_id'), json_extract(payload_json, '$.proposal.source'), json_extract(payload_json, '$.proposal.review_policy'), json_extract(payload_json, '$.proposal.current_plan_revision'), json_extract(payload_json, '$.proposal.plan_revision'), json_extract(payload_json, '$.proposal.preserved_prefix_count'), json_extract(payload_json, '$.proposal.supersedes_plan_proposal_id'), json_extract(payload_json, '$.decision.reason') from brain_events where event_type in ('planning.requested', 'planning.proposed', 'planning.adopted', 'planning.rejected') order by ts desc limit 30;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select proposed.event_id, proposed.ts, json_extract(proposed.payload_json, '$.proposal.plan_proposal_id'), adopted.event_id, adopted.event_type, repaired.event_id, repaired.event_type, json_extract(repaired.payload_json, '$.goal.goal_id') from brain_events proposed left join brain_events adopted on adopted.causal_parent_id = proposed.event_id and adopted.event_type = 'planning.adopted' left join brain_events repaired on repaired.causal_parent_id = adopted.event_id and repaired.event_type in ('goal.updated', 'goal.repaired') where proposed.event_type = 'planning.proposed' order by proposed.ts desc limit 20;"
```

Confirm these causal chains when applicable:

- `planning.requested -> planning.proposed -> planning.adopted -> goal.updated`
- `planning.requested -> planning.proposed -> planning.rejected`
- `planning.proposed -> planning.adopted -> goal.repaired`

## Manual scenario 6: why Blink did not adopt this plan

1. Trigger a proposal that contains an unknown capability or otherwise falls outside the bounded family policy.
2. Run the continuity audit.
3. Inspect `planning_digest`, then the raw `planning.rejected` event, then the downstream blocked goal state.

Expected outcome:

- the audit shows the rejected plan in `### Recent Rejected Plans`
- `planning_digest.reason_counts` explains the stable rejection reason
- the raw event chain shows `planning.requested -> planning.rejected`
- the goal / commitment state shows explicit bounded operator review instead of a silent planning dead end

## Pass criteria

A good run means:

- planning-needed work has an inspectable lifecycle
- adopted revisions are explicit and replayable
- review-required revisions stay review-bound
- wake-triggered work composes with the latest plan state correctly
- planning artifacts are at least as legible as autonomy and wake artifacts

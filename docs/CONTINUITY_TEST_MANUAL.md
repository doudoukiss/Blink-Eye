# Continuity Test Manual

This manual is the practical runbook for testing Blink's upgraded continuity
stack: memory, current-vs-historical truth, commitments, reflection,
autobiography, audit exports, and browser-first situational presence.

Use it when you want to validate the real behavior of the local brain, not
just whether the app boots.

## Scope

This runbook covers:

- typed memory and current-truth retrieval
- correction and supersession behavior
- durable commitments across restarts
- reflection and autobiographical updates
- operator continuity audits
- browser symbolic presence plus explicit `fetch_user_image` inspection
- browser camera stall, stale-frame, and recovery diagnostics

This runbook does not cover:

- robot-head live hardware bring-up
- telephony paths
- provider-specific cloud integrations

## Prerequisites

Recommended local baseline:

- Python `3.12`
- `uv`
- local Ollama running
- local model available, normally `qwen3.5:4b`
- `uv sync --python 3.12 --group dev` for the canonical automated proof lane

The automated brain-core proof lane is headless. It does not require browser,
audio, robot-head, provider, or local-runtime extras.

Quick smoke check:

```bash
uv run --python 3.12 blink-local-chat --once "Explain Blink in one sentence."
```

Primary browser path with MeloTTS and camera:

```bash
./scripts/run-local-browser-melo.sh
```

Primary English browser path with Kokoro and camera:

```bash
./scripts/run-local-browser-kokoro-en.sh
```

## Use an Isolated Brain DB

If you want a clean manual test run, point Blink at a fresh SQLite file before
starting the app:

```bash
mkdir -p artifacts/manual_test
export BLINK_LOCAL_BRAIN_DB_PATH="$PWD/artifacts/manual_test/brain.db"
```

If you do not override it, the default brain DB is:

```text
~/.cache/blink/brain/brain.db
```

## Test 1: Memory and Current Truth

Start Blink in text mode:

```bash
uv run --python 3.12 blink-local-chat --verbose
```

Then type:

1. `请记住：我叫 Alpha，我是产品工程师。`
2. `请记住我的偏好：我喜欢茉莉花茶。`
3. `我叫什么？我的职业是什么？我喜欢喝什么？`

Expected result:

- Blink recalls `Alpha`
- Blink recalls `产品工程师`
- Blink recalls `茉莉花茶`

### Correction and Supersession

Now type:

1. `更正一下：我其实不喜欢茉莉花茶，我更喜欢乌龙茶。`
2. `我现在喜欢喝什么？`
3. `我之前说过喜欢什么？`

Expected result:

- current truth is `乌龙茶`
- previous preference is still recoverable as historical truth
- the old claim is not silently deleted

## Test 2: Durable Commitments

In the same session, type:

1. `请记住一个任务：下周提醒我整理 Blink continuity audit。`
2. `我现在有哪些待办或承诺？`

Expected result:

- the task appears as an active commitment

Now restart Blink with the same DB and ask again:

1. stop Blink
2. start Blink again
3. ask `我现在还有哪些待办？`

Expected result:

- the commitment survives restart

Now complete it:

1. `把“整理 Blink continuity audit”标记为已完成。`
2. `我现在还有哪些待办？`

Expected result:

- the commitment is no longer active

## Test 3: Relationship Continuity and Reflection

Create a small run of related turns:

1. `我们最近一直在做 Blink 的 continuity architecture。`
2. `这个项目重点是 memory、reflection、commitment 和 audit。`
3. `请记住：我们当前的目标是把 continuity 做成可检查、可回放、可评估。`

Then run one bounded reflection cycle:

```bash
uv run --python 3.12 blink-local-brain-reflect --runtime-kind text
```

If you were testing through the browser runtime instead:

```bash
uv run --python 3.12 blink-local-brain-reflect --runtime-kind browser
```

Expected CLI output includes:

- `cycle_id=...`
- `draft_artifact=...`
- `health_report_id=...`

After reflection, ask Blink:

- `我们最近持续在做什么？`
- `你认为我们当前合作的主线是什么？`

Expected result:

- responses become more coherent across sessions
- relationship or project continuity should feel summarized rather than only
  turn-local

## Test 4: Operator Continuity Audit

Generate the continuity audit:

```bash
uv run --python 3.12 blink-local-brain-audit --runtime-kind text
```

Or for a browser session:

```bash
uv run --python 3.12 blink-local-brain-audit --runtime-kind browser
```

Artifacts are written under:

```text
artifacts/brain_audit/
```

Expected result:

- one JSON report
- one Markdown report

Inspect the audit for:

- `core_blocks`
- `current_claims`
- `historical_claims`
- `commitment_projection`
- `autonomy_ledger`
- `autonomy_digest`
- `reevaluation_digest`
- `wake_digest`
- `planning_digest`
- recent accepted actions
- recent suppressions / merges / non-actions
- non-action reevaluation conditions
- reevaluation trigger kinds and recent hold -> trigger -> outcome flows
- wake trigger counts / route counts / reason counts
- recent direct resumes / candidate proposals / keep-waiting decisions
- current pending plan proposals
- recent adopted plans / rejected plans / revision flows
- `relationship_arc`
- `health_report`
- `visual_health`
- reply/planning selection traces
- replay regression results

## Test 5: Browser Presence and Vision

Use either primary browser path for camera presence and grounded vision checks:

```bash
./scripts/run-local-browser-melo.sh
```

Open:

```text
http://127.0.0.1:7860/client/
```

With camera enabled:

1. wait 10 to 15 seconds
2. ask `你现在判断我在不在镜头前？我是在看镜头吗？`
3. then ask `详细看看摄像头里有什么`

Expected result:

- the first question uses lightweight symbolic situational state backed by the
  deterministic browser presence detector plus optional low-cadence enrichment
- the second triggers the explicit `fetch_user_image` tool path for
  higher-detail inspection
- continuous perception does not replace detailed visual inspection

Now test state change:

1. step out of frame
2. wait a few seconds
3. ask `你现在觉得我还在镜头前吗？`

Expected result:

- presence and engagement state updates
- Blink remains lightweight rather than narrating every frame

### Stale-Frame and Recovery Check

While the browser session is still running:

1. leave the tab open until the camera stream stalls or temporarily block the
   camera at the browser level
2. ask `你现在能确定我还在镜头里吗？`
3. run `uv run --python 3.12 blink-local-brain-audit --runtime-kind browser`
4. run `uv run --python 3.12 blink-local-doctor --profile browser --with-vision`

Expected result:

- Blink reports visual presence as uncertain instead of reusing an old
  confident answer
- the audit contains a `visual_health` section with track state, frame age,
  detector backend/confidence, and recovery attempts
- the doctor output reports the packaged presence detector plus the latest
  stored browser visual-health state

## Direct SQLite Inspection

These queries provide direct evidence of the upgraded continuity model.

If you exported `BLINK_LOCAL_BRAIN_DB_PATH`, use:

```bash
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_type, count(*) from brain_events group by 1 order by 2 desc;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select projection_name, scope_key, updated_at from brain_projections order by updated_at desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select block_kind, scope_type, scope_id, version, status, updated_at from core_memory_blocks order by updated_at desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select predicate, status, valid_from, valid_to, scope_type, scope_id from claims order by updated_at desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select commitment_id, title, status, goal_family, plan_revision, resume_count, updated_at from executive_commitments order by updated_at desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select entry_kind, status, salience, updated_at, rendered_summary from autobiographical_entries order by updated_at desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select report_id, status, score, created_at from memory_health_reports order by created_at desc limit 10;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_type, ts, json_extract(payload_json, '$.candidate_goal_id'), json_extract(payload_json, '$.reason') from brain_events where event_type in ('goal.candidate.created', 'goal.candidate.suppressed', 'goal.candidate.merged', 'goal.candidate.accepted', 'goal.candidate.expired', 'director.non_action.recorded') order by ts desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_type, ts, json_extract(payload_json, '$.trigger.kind'), json_extract(payload_json, '$.trigger.summary'), json_extract(payload_json, '$.candidate_goal_ids') from brain_events where event_type = 'director.reevaluation.triggered' order by ts desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_id, event_type, ts, json_extract(payload_json, '$.commitment.commitment_id'), json_extract(payload_json, '$.trigger.wake_kind'), json_extract(payload_json, '$.routing.route_kind'), json_extract(payload_json, '$.routing.details.reason') from brain_events where event_type = 'commitment.wake.triggered' order by ts desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select wake.event_id, wake.ts, json_extract(wake.payload_json, '$.commitment.commitment_id'), json_extract(wake.payload_json, '$.routing.route_kind'), resumed.event_id, json_extract(resumed.payload_json, '$.goal.goal_id') from brain_events wake left join brain_events resumed on resumed.causal_parent_id = wake.event_id and resumed.event_type = 'goal.resumed' where wake.event_type = 'commitment.wake.triggered' order by wake.ts desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select wake.event_id, wake.ts, json_extract(wake.payload_json, '$.commitment.commitment_id'), json_extract(wake.payload_json, '$.routing.route_kind'), candidate.event_id, json_extract(candidate.payload_json, '$.candidate_goal.candidate_goal_id') from brain_events wake left join brain_events candidate on candidate.causal_parent_id = wake.event_id and candidate.event_type = 'goal.candidate.created' where wake.event_type = 'commitment.wake.triggered' order by wake.ts desc limit 20;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select event_id, event_type, ts, json_extract(payload_json, '$.goal_id'), json_extract(payload_json, '$.commitment_id'), json_extract(payload_json, '$.proposal.plan_proposal_id'), json_extract(payload_json, '$.proposal.source'), json_extract(payload_json, '$.proposal.review_policy'), json_extract(payload_json, '$.proposal.current_plan_revision'), json_extract(payload_json, '$.proposal.plan_revision'), json_extract(payload_json, '$.proposal.preserved_prefix_count'), json_extract(payload_json, '$.proposal.supersedes_plan_proposal_id'), json_extract(payload_json, '$.proposal.missing_inputs'), json_extract(payload_json, '$.decision.reason') from brain_events where event_type in ('planning.requested', 'planning.proposed', 'planning.adopted', 'planning.rejected') order by ts desc limit 30;"
sqlite3 "$BLINK_LOCAL_BRAIN_DB_PATH" "select proposed.event_id, proposed.ts, json_extract(proposed.payload_json, '$.proposal.plan_proposal_id'), adopted.event_id, adopted.event_type, downstream.event_id, downstream.event_type, json_extract(downstream.payload_json, '$.goal.goal_id') from brain_events proposed left join brain_events adopted on adopted.causal_parent_id = proposed.event_id and adopted.event_type = 'planning.adopted' left join brain_events downstream on downstream.causal_parent_id = adopted.event_id and downstream.event_type in ('goal.updated', 'goal.repaired') where proposed.event_type = 'planning.proposed' order by proposed.ts desc limit 20;"
```

If you did not override the DB path, replace `$BLINK_LOCAL_BRAIN_DB_PATH` with:

```bash
~/.cache/blink/brain/brain.db
```

## Phase 6 / 7 / 8 / 9 Operator Review

When inspecting an autonomy, reevaluation, wake-router, or planning run, prefer this order:

1. check `autonomy_digest` in the thread digest or continuity audit
2. check `reevaluation_digest` for held work, triggers, and hold -> outcome chains
3. check `wake_digest` for waiting commitments, wake routes, and keep-waiting reasons
4. check `planning_digest` for pending proposals, adopted/rejected outcomes, and revision flows
5. confirm the raw `autonomy_ledger`, `commitment.wake.triggered`, and planning events match the digest summaries
6. inspect accepted goal ids, reevaluation conditions, wake causal chains, or planning causal chains for the specific case

For a clean operator read, confirm you can answer all seven:

- what candidate was current
- what Blink accepted, suppressed, merged, or held
- why Blink did nothing when it chose non-action
- what reevaluation trigger later resumed or expired the same candidate id
- what wake kind Blink matched for a deferred/blocked commitment
- why Blink resumed directly, proposed a candidate, or kept waiting
- what plan proposal is pending or current for the durable work
- why Blink adopted, rejected, or held that plan instead of silently mutating the tail

## Automated Regression Suite

Run the canonical automated brain-core proof lane first:

```bash
./scripts/test-brain-core.sh
```

Useful sub-lanes:

```bash
./scripts/test-brain-core.sh --lane fast
./scripts/test-brain-core.sh --lane proof
./scripts/test-brain-core.sh --lane fuzz-smoke
```

Then use the focused suites below when you need phase-specific investigation.

Targeted continuity suite:

```bash
uv run pytest \
  tests/test_brain_context_policy.py \
  tests/test_brain_continuity_evals.py \
  tests/test_brain_audit_reports.py \
  tests/test_brain_replay.py \
  tests/test_brain_runtime.py \
  tests/test_brain_memory_v2.py \
  tests/test_brain_reflection.py \
  tests/test_brain_commitments.py -q
```

### Phase 6 Automatic Coverage

For the Presence Director and bounded initiative layer specifically, run:

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

What this automatic lane proves:

- perception, commitment-wake, and maintenance producers emit bounded candidate
  work
- Presence Director policy records `suppressed`, `merged`, `accepted`, and
  `non_action` outcomes replayably
- capability-family routing stays bounded and inspectable
- audit and replay artifacts keep action and non-action trails legible

### Phase 7 Automatic Coverage

For reevaluation, expiry cleanup, and operator-facing digest coverage, run:

```bash
uv run pytest \
  tests/test_brain_autonomy.py \
  tests/test_brain_replay.py \
  tests/test_brain_audit_reports.py \
  tests/test_brain_layered_memory.py \
  tests/test_brain_runtime.py \
  tests/test_brain_reflection.py -q
```

What this automatic lane proves:

- held candidates can continue through reevaluation without duplicate proposals
- replay fixtures cover hold -> reevaluate -> accept and hold -> reevaluate -> expire
- `reevaluation_digest` stays deterministic across repeated replays
- audit JSON and Markdown keep reevaluation triggers and outcomes operator-readable

### Phase 8 Automatic Coverage

For the wake router and operator-facing wake reports, run:

```bash
uv run pytest \
  tests/test_brain_commitments.py \
  tests/test_brain_replay.py \
  tests/test_brain_audit_reports.py \
  tests/test_brain_layered_memory.py -q
```

What this automatic lane proves:

- replay fixtures cover all three landed wake outcomes:
  `resume_direct`, `propose_candidate`, and `keep_waiting`
- `wake_digest` stays deterministic across repeated replays
- audit JSON and Markdown keep action and non-action wake outcomes equally legible
- raw wake events remain causally inspectable through
  `commitment.wake.triggered -> goal.resumed` and
  `commitment.wake.triggered -> goal.candidate.created`

### Phase 9 Automatic Coverage

For bounded deliberation and plan-revision operator coverage, run:

```bash
uv run pytest \
  tests/test_brain_planning.py \
  tests/test_brain_context_policy.py \
  tests/test_brain_replay.py \
  tests/test_brain_audit_reports.py \
  tests/test_brain_layered_memory.py \
  tests/test_brain_import_hygiene.py -q
```

What this automatic lane proves:

- replay fixtures cover `propose -> adopt`, `propose -> reject`, and `revise-after-block -> adopt`
- `planning_digest` stays deterministic across repeated replays
- audit JSON and Markdown keep adopted, rejected, and pending review-bound plan outcomes equally legible
- raw planning events remain causally inspectable through
  `planning.requested -> planning.proposed -> planning.adopted -> goal.updated`,
  `planning.requested -> planning.rejected`, and
  `planning.proposed -> planning.adopted -> goal.repaired`

Optional browser smoke:

```bash
uv run python -m playwright install chromium
BLINK_RUN_BROWSER_E2E=1 uv run pytest tests/test_browser_e2e.py -q
```

What the browser smoke proves:

- the Blink browser UI loads
- the client reaches the live SmallWebRTC runtime
- fake media can establish a browser session

What it does not prove:

- real `person_present=present` camera detection
- real scene-change candidate production from a human re-entry
- the narrow spoken re-engagement path under real turn timing

Use the manual browser QA runbook in
[AUTONOMOUS_PRESENCE_DIRECTOR_QA.md](./AUTONOMOUS_PRESENCE_DIRECTOR_QA.md)
for those final human-camera checks.

## Pass Criteria

A good run means:

- remembered facts persist across restart
- corrections replace current truth without erasing history
- tasks survive restart and can be completed cleanly
- reflection produces a draft artifact and improves continuity summaries
- audit exports JSON plus Markdown with claims, commitments, arcs, health, and
  selection traces
- browser presence updates without replacing explicit detailed vision
- stale camera frames degrade to `uncertain` instead of being reused as fresh observations
- detailed camera inspection still goes through `fetch_user_image`

## Related Docs

- [LOCAL_DEVELOPMENT.md](../LOCAL_DEVELOPMENT.md)
- [USER_MANUAL.md](./USER_MANUAL.md)
- [Roadmap](./roadmap.md)

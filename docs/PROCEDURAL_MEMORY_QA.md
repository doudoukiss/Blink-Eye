# Procedural Memory QA

Use this runbook to validate the **Phase 11 procedural-memory and skill
consolidation** stack with replay-safe evidence first, and live demos second.

## Validation ladder

Use this order. Do not treat later steps as a substitute for earlier ones.

1. Checked-in replay fixtures are the canonical proof surface.
2. Focused pytest lanes validate replay, eval, audit, layered export, and import hygiene.
3. Audit artifacts provide the operator-readable and machine-readable inspection surface.
4. Optional live runtime validation is a follow-on sanity check only.

## Canonical replay fixtures

Use these checked-in fixtures under `tests/fixtures/brain_evals/`:

- Learning: `procedural_skill_candidate_then_active.json`
- Reuse: `planning_skill_reuse_exact.json`
- Negative transfer: `planning_skill_reject_mismatch.json`
- Retirement: `procedural_skill_retired_after_repeated_failures.json`
- Supersession: `procedural_skill_superseded_by_revised_tail.json`
- Delta reuse: `planning_skill_delta_revise_tail.json`

These fixtures carry `qa_categories` and are the primary evidence that the same
event stream rebuilds the same procedural artifacts.

## Focused test lanes

Run these focused lanes before relying on any manual interpretation:

```bash
uv run pytest \
  tests/test_brain_replay.py \
  tests/test_brain_continuity_evals.py \
  tests/test_brain_audit_reports.py \
  tests/test_brain_layered_memory.py \
  tests/test_brain_import_hygiene.py \
  tests/test_brain_planning.py \
  tests/test_brain_context_policy.py -q
```

Minimum expectations:

- replay fixtures pass deterministically
- scripted eval cases pass for learning, reuse, negative transfer, retirement, and supersession
- replay fixtures and scripted evals rebuild the same procedural planning state from the same
  event stream
- audit JSON and Markdown include procedural governance and QA surfaces
- layered-memory thread exports include empty/default procedural QA surfaces before any skill evidence exists
- blocked-import collection still succeeds without `openai` or `pyloudnorm`

Important determinism note:

- fixture-backed evals must preserve original `event_id` and `ts` when rehydrating replay cases,
  otherwise procedural step/result linkage will drift and failure signatures will under-report

## Audit artifact inspection

Generate an audit when you need an operator-facing review surface:

```bash
uv run --python 3.12 blink-local-brain-audit \
  --brain-db-path /path/to/brain.db \
  --output-dir /tmp/blink-procedural-audit \
  --replay-cases-dir tests/fixtures/brain_evals
```

Inspect these JSON fields:

- `procedural_skills.active_skill_ids`
- `procedural_skills.candidate_skill_ids`
- `procedural_skills.retired_skill_ids`
- `procedural_skills.superseded_skill_ids`
- `procedural_skill_digest.recent_supersessions`
- `procedural_skill_digest.recent_retirements`
- `procedural_skill_digest.top_failure_signatures`
- `procedural_skill_governance_report.low_confidence_skill_ids`
- `procedural_skill_governance_report.retirement_reason_counts`
- `procedural_skill_governance_report.negative_transfer_reason_counts`
- `procedural_skill_governance_report.follow_up_trace_ids`
- `planning_digest.procedural_origin_counts`
- `planning_digest.skill_rejection_reason_counts`
- `planning_digest.delta_operation_counts`
- `context_packet_digest.planning.selected_anchor_types`
- `context_packet_digest.planning.selected_backing_ids`
- `procedural_qa_report.case_counts`
- `procedural_qa_report.coverage_flags`
- `procedural_qa_report.failed_case_names`
- `procedural_qa_report.procedural_origin_counts`
- `procedural_qa_report.skill_rejection_reason_counts`
- `procedural_qa_report.delta_operation_counts`
- `procedural_qa_report.selected_skill_ids`
- `procedural_qa_report.negative_transfer_reason_counts`
- `procedural_qa_report.rejected_reusable_skill_ids`
- `procedural_qa_report.high_risk_failure_signature_codes`
- `procedural_qa_report.follow_up_trace_ids`
- `procedural_qa_report.recent_negative_transfer_flows`
- `procedural_qa_report.recent_skill_flows`

Inspect these Markdown sections:

- `## Procedural Review`
- `### Active Skills`
- `### Candidate Skills`
- `### Retired / Superseded Skills`
- `### Failure Signatures`
- `## Procedural QA Review`
- `### Learning Lifecycle`
- `### Skill Reuse / Delta`
- `### Negative Transfer / Rejections`
- `### Retirement / Supersession`

Operator questions the artifacts must answer clearly:

- Which skill was learned or reused?
- Which traces and outcomes support it?
- Was the plan fresh, exact reuse, or bounded delta?
- Which prior skill was rejected, and why?
- Which failures lowered confidence or triggered retirement?
- Which skill superseded an older procedure?
- Does the replay-backed QA report agree with the live procedural inventory and planning digest?

Concrete strings and ids you should expect to see in the replay-backed QA lane:

- `skill_reuse_mismatch` in `procedural_qa_report.negative_transfer_reason_counts`
- selected procedural skill ids under `procedural_qa_report.selected_skill_ids`
- high-risk failure codes like `failed` or `operator_review` under
  `procedural_qa_report.high_risk_failure_signature_codes`

## Manual runtime follow-on

Only do this after the replay and focused test evidence is green.

Recommended follow-on checks:

1. Create one repeated bounded task and confirm a later run surfaces skill reuse.
2. Change a key condition and confirm the prior skill is rejected or narrowly adapted.
3. Reproduce repeated relevant failures and confirm the skill becomes low-confidence or retired.
4. Export an audit and verify the same story appears in:
   - `planning_digest`
   - `procedural_skill_governance_report`
   - `procedural_qa_report`
5. Confirm the audit markdown shows replay-backed procedural evidence even if the live thread itself
   has no local negative-transfer event history.

## Pass / fail checklist

- [ ] Replay fixtures pass for learning, reuse, negative transfer, retirement, and supersession.
- [ ] Procedural confidence is explicit in skill/governance artifacts.
- [ ] Retirement reasons are explicit in skill/governance artifacts.
- [ ] Negative transfer is visible as first-class rejection behavior, not hidden in generic planning logs.
- [ ] `procedural_qa_report` carries replay-backed selected skills, rejection reasons, delta counts,
      and follow-up trace ids.
- [ ] Audit markdown shows replay-backed procedural QA evidence such as `skill_reuse_mismatch`.
- [ ] Procedural artifacts remain deterministic after replay.
- [ ] Import-hygiene coverage remains green for focused procedural/continuity suites.
- [ ] Live runtime checks are consistent with the replay-safe artifact story.

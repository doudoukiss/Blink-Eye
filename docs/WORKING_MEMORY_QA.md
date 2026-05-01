# Active Situation Model / Working Memory QA

Use this runbook after changing any Phase 12 working-memory or situation-model behavior.

## What this phase is about

The phase is about making active internal state explicit and inspectable:

- what Blink currently believes
- what is still uncertain
- what has gone stale
- what private task state should persist across interruptions or branches

It is **not** about hiding more state from operators.

## Minimum proof before manual QA

1. Focused replay / proof suites pass.
2. Working-memory and situation digests render in audit output.
3. Runtime-backed tests that do not require heavy audio deps still collect cleanly.
4. The deterministic Atheris harness smoke lane stays green.

Recommended minimum commands:

```bash
uv run pytest tests/brain_properties -q
uv run pytest tests/brain_stateful -q
uv run pytest tests/test_brain_fuzz_harness_smoke.py -q
uv run pytest tests/test_brain_import_hygiene.py::test_lower_brain_collection_stays_provider_light -q
```

Inspect these fields first:

- `private_working_memory.active_record_ids`
- `private_working_memory.stale_record_ids`
- `private_working_memory.resolved_record_ids`
- `private_working_memory_digest.unresolved_record_ids`
- `active_situation_model.active_record_ids`
- `active_situation_model.stale_record_ids`
- `active_situation_model.unresolved_record_ids`
- `active_situation_model_digest.kind_counts`
- `active_situation_model_digest.state_counts`
- `active_situation_model_digest.uncertainty_code_counts`
- `active_situation_model_digest.linked_commitment_ids`
- `active_situation_model_digest.linked_plan_proposal_ids`
- `active_situation_model_digest.linked_skill_ids`

## Manual scenarios

### Scenario 1: interrupted task continuity

1. Start a bounded multi-step task.
2. Interrupt it with an unrelated user turn.
3. Return to the task.
4. Confirm Blink preserves relevant active assumptions without pretending stale ones are current.

Expected:

- active goal / plan context remains visible
- unrelated turn chatter does not fully erase relevant state
- stale scene assumptions are not treated as current truth

### Scenario 2: stale scene invalidation

1. Seed a scene-linked assumption.
2. Let the scene age past freshness or inject contradictory observation.
3. Ask Blink to act on the old assumption.

Expected:

- Blink marks the assumption stale or uncertain
- Blink defers, refreshes, or asks for observation instead of acting as if nothing changed

### Scenario 3: contradiction remains explicit

1. Create two conflicting user / scene signals.
2. Ask a question that depends on the conflict.

Expected:

- the conflict remains visible in digests or packet trace
- Blink does not flatten both signals into one confident summary

### Scenario 4: branch consistency

1. Create active private task state.
2. Fork the conversation into two branches.
3. Ensure each branch remains internally coherent.

Expected:

- relevant branch-local active state stays consistent
- hidden state does not leak incorrectly across incompatible branches

## Things operators should be able to answer

- what was the active goal / commitment / plan assumption
- what scene state was considered current
- what evidence supported the active state
- what was uncertain or stale
- why Blink deferred or requested refresh

## Failure smells

- “it worked in the demo” but active state is not reconstructible from replay
- compiler output silently carries stale scene assumptions forward
- branch / interruption behavior looks plausible but cannot be explained from artifacts
- working-memory buffers grow without a clear bound
- optional-link ids like `None` or blank text survive parsing as fake live links

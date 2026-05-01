# Narrow Atheris Harnesses

These harnesses are opt-in and intentionally stay outside normal pytest
collection. They only target compact parser and replay-ingestion surfaces.

## Run

```bash
uv run --with atheris python tests/fuzz/atheris/fuzz_graph_projection.py -atheris_runs=1000
uv run --with atheris python tests/fuzz/atheris/fuzz_dossier_projection.py -atheris_runs=1000
uv run --with atheris python tests/fuzz/atheris/fuzz_procedural_records.py -atheris_runs=1000
uv run --with atheris python tests/fuzz/atheris/fuzz_replay_artifacts.py -atheris_runs=500
uv run --with atheris python tests/fuzz/atheris/fuzz_active_state_projections.py -atheris_runs=1000
uv run --with atheris python tests/fuzz/atheris/fuzz_core_projection_records.py -atheris_runs=1000
uv run --with atheris python tests/fuzz/atheris/fuzz_active_state_digests.py -atheris_runs=1000
uv run --with atheris python tests/fuzz/atheris/fuzz_reevaluation_digests.py -atheris_runs=1000
```

To use an explicit corpus, pass one or more directories before the Atheris
flags:

```bash
uv run --with atheris python tests/fuzz/atheris/fuzz_graph_projection.py /tmp/blink-graph-corpus -atheris_runs=1000
```

## Seed Shapes

Graph projection seed:

```json
{
  "scope_type": "user",
  "scope_id": "user-1",
  "nodes": [
    {
      "node_id": "node-1",
      "kind": "claim",
      "backing_record_id": "claim-1",
      "summary": "likes tea",
      "status": "current"
    }
  ],
  "edges": []
}
```

Dossier projection seed:

```json
{
  "scope_type": "user",
  "scope_id": "user-1",
  "dossiers": [
    {
      "dossier_id": "dossier-1",
      "kind": "relationship",
      "scope_type": "user",
      "scope_id": "user-1"
    }
  ]
}
```

Procedural record seed:

```json
{
  "trace_id": "trace-1",
  "goal_id": "goal-1",
  "plan_proposal_id": "proposal-1",
  "goal_family": "conversation",
  "status": "completed"
}
```

Replay artifact seed:

```json
{
  "session": {
    "agent_id": "agent-1",
    "user_id": "user-1",
    "session_id": "session-1",
    "thread_id": "thread-1"
  },
  "events": [
    {
      "event_type": "memory.claim.recorded",
      "payload": {
        "subject_entity_id": "user-1",
        "predicate": "preference.like",
        "object": {"value": "tea"}
      }
    }
  ]
}
```

## Notes

- `_structured_inputs.py` first tries UTF-8 JSON, then falls back to a bounded
  `atheris.FuzzedDataProvider` tree.
- The replay harness uses `BrainStore(path=":memory:")` and only exercises
  append/materialization helpers plus the compact cross-phase operator digests,
  not the full runtime.
- The active-state and reevaluation harnesses stay on pure parser and digest
  seams; they do not bootstrap browser, audio, provider, or full runtime stacks.
- Safe rejection is intentionally narrow: malformed coercions may return early,
  but unexpected parser or projection exceptions still fail the harness.
- On stock macOS Apple Clang environments, `atheris` may fail to build because
  `libFuzzer` is unavailable. Use
  `uv run pytest tests/test_brain_fuzz_harness_smoke.py -q`
  as the deterministic per-PR fallback that stubs `atheris`, imports every
  harness, and exercises `TestOneInput(...)` on compact JSON seeds.

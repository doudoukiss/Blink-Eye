# Fuzz Test Quickstart

This guide is intentionally practical.

Use it when you want to add a real proof lane for Blink without coupling tests
to unrelated runtime/audio/provider stacks.

## Recommended tool split

### Use Hypothesis for most Blink brain tests

Use Hypothesis when you want to test:

- invariants over many valid inputs
- replay determinism
- graph / dossier / packet properties
- private working memory / active situation / scene world invariants
- bounded plan / skill invariants
- valid multi-step operation sequences

### Use Atheris only for narrow surfaces

Use Atheris when you want to stress:

- import/export loaders
- `from_dict()` / `as_dict()` roundtrips
- compact pure normalization helpers
- replay artifact ingestion
- compact digest builders for autonomy and active-state summaries

Do **not** start by pointing Atheris at the full runtime.

## Recommended test folders

```text
tests/brain_properties/
tests/brain_stateful/
tests/fuzz/atheris/
```

## Suggested shared Hypothesis profile

Register one lightweight default profile in `tests/conftest.py` or a helper
module loaded by pytest:

- smaller `max_examples` for per-PR runs
- explicit `stateful_step_count` for state machines
- controlled or disabled deadlines where CI timing is noisy
- explicit seeding only for reproducing failures, not for normal coverage

The repo currently uses:

- `brain_fast` as the default profile
- `brain_stateful` for bounded rule-based machines

## What to run often

### Per-PR

```bash
./scripts/test-brain-core.sh
./scripts/test-brain-core.sh --lane fast
./scripts/test-brain-core.sh --lane proof
./scripts/test-brain-core.sh --lane fuzz-smoke
```

### Nightly or opt-in

```bash
./scripts/test-brain-core.sh --lane proof
./scripts/test-brain-core.sh --lane atheris
```

If `atheris` cannot build on macOS because the local Apple Clang toolchain does
not ship `libFuzzer`, keep the deterministic smoke lane in PR coverage and run
the real Atheris commands only on libFuzzer-capable machines.

On macOS, the usual fix is to install Homebrew LLVM and run Atheris with that
toolchain on `PATH`.

## Best first properties

### Replay / projection
- same normalized event stream -> same rebuilt state
- replay artifacts roundtrip cleanly
- duplicate irrelevant noise does not mutate core outcome

### Graph / dossiers
- current and historical node partitions stay disjoint
- contradiction or supersession prevents stale facts from remaining current
- dossier evidence refs always point to reachable graph/entity/claim state

### Context compiler
- packet budget is never exceeded
- support trace ids are subsets of reachable source evidence
- selector order changes do not violate required packet contracts
- degraded active-state packets keep their bounded required sections

### Planning / procedural
- reused skill ids must have supporting traces
- rejected reuse must surface a reason
- bounded delta stays bounded

### Active state / reevaluation
- private working memory records stay capped and provenance-linked
- scene-world freshness only decays without newer evidence
- active-situation links never point to missing private/planning/procedural state
- presence director reevaluation never bypasses explicit hold policy

## Common mistakes

### 1. Mutating pytest fixtures inside Hypothesis tests
Prefer factories or local setup inside the test body.
Do not rely on mutable shared fixtures for per-example isolation.

### 2. Using snapshots where you need invariants
Property tests should assert structural truths, not freeze huge payload blobs.

### 3. Fuzzing the full runtime first
Start with headless brain-core surfaces.

### 4. Treating flaky reruns as proof
Reruns can triage noise, but they do not replace invariant testing.

## Reproducing failures

When a property or stateful test fails:

1. save the minimized failing example
2. add an explicit regression test or targeted example
3. keep the property test
4. record whether the bug was:
   - replay drift
   - stale/current mixup
   - packet-budget overflow
   - provenance loss
   - invalid skill reuse
   - import/dependency leak

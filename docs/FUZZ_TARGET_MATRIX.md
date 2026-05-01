# Fuzz Target Matrix

This matrix is for selecting worthwhile proof targets.

## 1. Event store / replay / projections

### Paths
- `src/blink/brain/core/store.py`
- `src/blink/brain/core/replay.py`
- `src/blink/brain/core/projections.py`
- `src/blink/brain/replay.py`
- `src/blink/brain/executive_policy_audit.py`
- `src/blink/brain/runtime_shell_digest.py`

### Best tool
- Hypothesis properties
- Hypothesis state machines for event sequences
- optional Atheris for artifact import/export loaders

### Core invariants
- normalized equivalent streams rebuild identical state
- replay does not invent new current state
- malformed artifact data fails safely rather than corrupting projections
- exported artifacts can roundtrip through loaders
- cross-phase operator digests remain JSON-safe and deterministic under compact
  malformed inputs
- executive-policy and runtime-shell summaries remain derivable from compact
  replay/audit inputs without bootstrapping live runtime state

---

## 2. Continuity graph

### Paths
- `src/blink/brain/memory_v2/graph.py`

### Best tool
- Hypothesis properties
- optional Atheris for projection parsing

### Core invariants
- current / historical / superseded partitions remain coherent
- edge references point to existing nodes
- node and edge counts remain non-negative and bounded by inputs
- contradiction / supersession transitions preserve provenance

---

## 3. Dossiers

### Paths
- `src/blink/brain/memory_v2/dossiers.py`

### Best tool
- Hypothesis properties
- optional Atheris for record parsing

### Core invariants
- freshness flags match evidence timing state
- contradicted or uncertain dossiers surface in the right projection sets
- evidence refs stay structurally valid
- summary facts do not outlive their backing evidence state

---

## 4. Context compiler

### Paths
- `src/blink/brain/context/compiler.py`
- `src/blink/brain/context/selectors.py`
- `src/blink/brain/context/budgets.py`
- `src/blink/brain/context_surfaces.py`

### Best tool
- Hypothesis properties
- seed-based packet sweeps

### Core invariants
- packet budget never exceeds configured ceiling
- compilation trace is internally consistent
- selected trace ids refer to reachable source records
- required sections for `reply` and `planning` packets are present
- degraded active-state packets keep bounded `active_state` / `unresolved_state` coverage

---

## 5. Active state stack

### Paths
- `src/blink/brain/private_working_memory.py`
- `src/blink/brain/active_situation_model.py`
- `src/blink/brain/scene_world_state.py`
- `src/blink/brain/private_working_memory_digest.py`
- `src/blink/brain/active_situation_model_digest.py`
- `src/blink/brain/scene_world_state_digest.py`

### Best tool
- Hypothesis properties
- seed-based packet sweeps
- optional Atheris for projection parsers and digests

### Core invariants
- equivalent normalized inputs rebuild identical active-state projections
- per-buffer / per-kind / zone / entity / affordance caps always hold
- source ids, backing ids, and cross-links stay reachable
- freshness only decays without newer evidence
- degraded mode and reason codes stay coherent with disconnected or stale perception

---

## 6. Planning / revision / wake

### Paths
- `src/blink/brain/_executive/planning.py`
- `src/blink/brain/_executive/wake_router.py`
- `src/blink/brain/executive.py`
- `src/blink/brain/_executive/presence_director.py`
- `src/blink/brain/runtime.py`

### Best tool
- Hypothesis rule-based state machines
- small Hypothesis properties for runtime alarm selection

### Core invariants
- adopted plans have explicit provenance
- rejected plans expose reason codes
- revision preserves boundedness constraints
- wake-triggered resume never bypasses capability rules
- reevaluation only resumes candidates when the hold kind actually matches
- runtime reevaluation alarms track only the nearest pending wake

---

## 7. Procedural traces and skills

### Paths
- `src/blink/brain/memory_v2/procedural.py`
- `src/blink/brain/memory_v2/skills.py`
- `src/blink/brain/procedural_planning.py`

### Best tool
- Hypothesis properties
- Hypothesis rule-based state machines
- optional Atheris for import/export records

### Core invariants
- active skills have support traces
- retired/superseded relations are coherent
- negative transfer reduces confidence or blocks reuse
- selected skill ids are compatible with planner decision details

---

## 8. Import hygiene / degraded mode

### Paths
- `tests/test_brain_import_hygiene.py`
- `tests/brain_core/test_import_hygiene.py`
- `src/blink/brain/runtime.py`

### Best tool
- deterministic example tests
- targeted environment-missing shims
- not Atheris

### Core invariants
- brain-core tests do not unexpectedly import heavy runtime/audio stacks
- optional dependency absence degrades gracefully
- proof lanes stay runnable on minimal local setups
- deterministic harness-smoke coverage keeps Atheris modules healthy on machines
  that cannot build `libFuzzer`

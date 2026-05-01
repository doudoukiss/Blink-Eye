# 07 — Rollout and release gate

## Rollout model

Each phase should land behind a feature flag or safe default. The 12-phase
bilingual actor runtime is now implemented as an additive default-on browser
surface with compatibility endpoints preserved.

Current operator controls:

```text
BLINK_LOCAL_BROWSER_VISION=1|0
BLINK_LOCAL_CONTINUOUS_PERCEPTION=0|1
BLINK_LOCAL_ALLOW_BARGE_IN=0|1
BLINK_LOCAL_ACTOR_SURFACE_V2=1|0
BLINK_LOCAL_ACTOR_TRACE=0|1
BLINK_LOCAL_ACTOR_TRACE_DIR=artifacts/actor_traces
```

The v2 actor event/state contracts, conversation floor, active listener,
camera scene state, persona/memory compiler surfaces, and interruption policy
are not separate experimental toggles. They are the public browser runtime
contract for both primary profiles. Roll back through compatibility adapters or
the actor surface UI flag, not by silently changing one profile's schema.

## Required validation after each phase

```bash
uv run ruff check src/blink/interaction src/blink/brain src/blink/cli/local_browser.py tests
uv run --extra runner --extra webrtc pytest tests/test_local_workflows.py
uv run --extra runner --extra webrtc pytest tests/test_*actor*.py tests/test_*browser*.py
```

Add targeted tests as phases introduce modules.

For actor-runtime changes, also run:

```bash
./scripts/eval-bilingual-actor-bench.sh
```

## Manual dogfooding minimum

At least three five-minute sessions:

1. Long technical planning turn.
2. Camera object/showing query.
3. Interruption and correction session.

Each session should save an actor trace and a human rating form.

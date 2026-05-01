# Blink Bilingual Actor Runtime

This folder documents the implemented bilingual browser actor runtime. The
runtime makes Blink's listening, thinking, looking, speaking, interruption,
memory/persona use, and degradation state visible through public-safe browser
APIs and deterministic release gates.

This is the part of Blink most directly shaped by the GenJen paper set. Instead
of treating the browser as a thin chat frontend, Blink treats it as an actor
runtime: a live system with speaking/listening streams, online state,
interruption boundaries, camera honesty, persona references, and release gates
for perceived interaction quality.

The design goal is advanced but concrete. Blink does not ship a realistic video
avatar; it builds the substrate a serious real-time assistant needs before any
avatar, robot, or hosted provider is allowed to sit on top.

## Primary Browser Paths

Blink has two equal primary local browser product paths:

| Path | Profile | Language | TTS | Vision | Continuous perception |
| --- | --- | --- | --- | --- | --- |
| `./scripts/run-local-browser-melo.sh` | `browser-zh-melo` | `zh` | `local-http-wav/MeloTTS` | Moondream on by default | Off by default |
| `./scripts/run-local-browser-kokoro-en.sh` | `browser-en-kokoro` | `en` | `kokoro/English` | Moondream on by default | Off by default |

Both use the same client URL:

```text
http://127.0.0.1:7860/client/
```

Use `--no-vision` or `BLINK_LOCAL_BROWSER_VISION=0` only when intentionally
testing an audio-only browser run. Use `--continuous-perception` or
`BLINK_LOCAL_CONTINUOUS_PERCEPTION=1` only for explicit low-frequency scene
updates.

## Public Runtime Surfaces

- `/api/runtime/actor-state`: schema-v2 live actor snapshot for the browser UI
  and diagnostics.
- `/api/runtime/actor-events`: schema-v2 public-safe actor event polling.
- `/api/runtime/performance-state`: schema-v1 compatibility state.
- `/api/runtime/performance-events`: schema-v1 compatibility events.
- `/api/runtime/client-config.js`: browser feature flags, including actor
  surface v2.
- `scripts/evals/replay-actor-trace.py`: offline replay for saved actor traces.
- `./scripts/eval-bilingual-actor-bench.sh`: deterministic release gate for
  both primary profiles.

Persistent actor traces are opt-in. Enable them with `BLINK_LOCAL_ACTOR_TRACE=1`
or `--actor-trace`; default traces store sanitized event records only.

## Performance Intelligence V3 Baseline

The Bilingual Performance Intelligence V3 upgrade builds on this actor runtime;
it does not replace the public actor-state, actor-events, compatibility
performance endpoints, browser actor surface, or bilingual actor release gate.

Phase 01 locks a deterministic machine-readable baseline at:

```text
evals/bilingual_performance_intelligence_v3/baseline_profiles.json
```

Regenerate or check it without opening the browser or calling models:

```bash
./scripts/eval-performance-intelligence-baseline.sh
./scripts/eval-performance-intelligence-baseline.sh --check
```

The baseline records both primary browser paths with equal product status,
Moondream/browser vision enabled by default, continuous perception disabled by
default, protected playback enabled by default, and the public actor runtime
endpoints that future V3 layers must preserve.

Phase 02 adds opt-in `PerformanceEpisodeV3` ledgers above the actor event
stream. Enable live episode JSONL with `BLINK_LOCAL_PERFORMANCE_EPISODE_V3=1`
or `--performance-episode-v3`; artifacts default to
`artifacts/performance_episodes_v3/` and store hashes, counts, labels, reason
codes, and segment timings rather than raw transcripts, media, prompts, SDP, or
memory bodies.

Replay or convert offline without browser, model, camera, TTS, or audio
services:

```bash
uv run python scripts/evals/replay-performance-episode-v3.py episode.jsonl
uv run python scripts/evals/replay-performance-episode-v3.py actor-trace.jsonl \
  --input-format actor-trace --output-episodes episode.jsonl
```

Phase 03 adds an internal `ActorControlFrameV3` scheduler above actor events.
It keeps the public browser endpoints stable, applies deterministic control
frames only at VAD, STT-final, speech-chunk, TTS-queue, camera-frame,
tool-result, interruption, and repair boundaries, and caps live speech/subtitle
lookahead so interruptions are not hidden behind a long queued response. Replay
and conversion stay offline:

```bash
uv run python scripts/evals/replay-actor-control-frame-v3.py actor-trace.jsonl \
  --input-format actor-trace --output-control-frames control.jsonl
uv run python scripts/evals/replay-actor-control-frame-v3.py control.jsonl
```

Phase 04 extends the existing conversation floor controller with a floor-v3
policy model for overlap, ignored backchannels, explicit interruption, repair,
and handoff sub-states. The public `conversation_floor` object keeps
`schema_version: 1` for compatibility and adds `floor_model_version: 3`,
public-safe phrase class/confidence labels, yield decisions, and WebRTC
echo-policy labels. Native PyAudio paths remain backend isolation lanes; the
primary product UX lanes are still the two browser/WebRTC paths above.

Phase 05 adds `SemanticListenerStateV3` inside the existing
`active_listening.schema_version=2` object. The browser UI can show localized
listener chips for heard summary, constraints, questions, object showing,
continued listening, and answer readiness while the user speaks. This remains a
visual listener surface only: it does not add audible backchannels, does not add
new HTTP endpoints, and persistent traces/episodes/control frames keep only
semantic labels, chip IDs, hashes, counts, and reason codes. Camera-related
chips require fresh camera-scene evidence; stale or unsupported camera state is
reported as limited rather than visual understanding.

Phase 06 adds `SceneSocialStateV2` inside the existing `camera_scene` object.
It makes camera honesty explicit with `can_see_now`, `recent_frame_available`,
`available_not_used`, and `unavailable`, and replays scene transitions such as
`looking_requested`, `frame_captured`, `vision_answered`, `vision_stale`, and
`vision_unavailable`. It remains an additive scene-state layer over on-demand
Moondream grounding; continuous perception stays off by default and no public
trace stores raw images or raw Moondream text.

Phase 07 adds `PerformancePlanV3` beside the existing `performance_plan_v2`
inside the memory/persona performance surface. The V3 plan compiles the latest
internal ActorControl frame, semantic listener labels, scene-social camera
honesty, memory continuity, persona references, and TTS capability declarations
into a bounded public plan summary for each turn. It changes only supported
behavioral controls such as chunk budget, subtitle policy, camera-reference
honesty, repair stance, and memory callback policy; MeloTTS and Kokoro still do
not claim speech-rate, emotional prosody, stream-abort, hardware, or fake-human
controls.

Phase 08 extends the dual TTS speech director contract with
`speech_director_version: 3` metadata. MeloTTS and Kokoro chunks now carry
deterministic duration estimates, subtitle timing policy, stale-generation
state, and backend capability labels through existing actor events, control
frames, and episode replay. The services themselves remain unchanged: subtitles
are emitted before or at playback start, chunk budgets stay bounded by the
backend defaults, and unsupported rate/prosody/pause/hardware controls remain
explicit no-ops. Browser-created TTS services use chunk-aligned audio contexts
so every V3 speech chunk has its own TTS queue boundary; native PyAudio lanes
keep their backend-isolation defaults.

Phase 09 adds `PersonaReferenceBankV3` style anchors. Blink now cites selected
public behavior anchors in `PerformancePlanV3` by situation key, so
interruption repair, correction, visual grounding, uncertainty, memory
callback, disagreement, planning, casual check-in, and bounded playfulness are
inspectable without exposing hidden prompts or persona prose. The existing
operator workbench renders the public anchor catalog; no browser runtime
endpoint is added or replaced.

Phase 10 adds `DiscourseEpisode` records and a nested `memory_continuity_v3`
trace. Performance episodes now derive typed memory cues such as active project,
preference, correction, visual event, repeated frustration, and success pattern;
the brain store persists them as `memory.discourse_episode.derived` events, not
a new table. PerformancePlanV3 consumes those cues to make callbacks behavior
changing, cross-language, conflict-aware, and stale-memory-safe while public
surfaces expose only IDs, summaries, labels, counts, and reason codes.

Phase 11 adds a local performance-learning flywheel above the same browser
runtime. The operator workbench can review public-safe episode evidence, rate
candidate pairs across fixed dimensions, append `PerformancePreferencePair`
JSONL, and generate evidence-cited `PerformanceLearningPolicyProposalV3`
records. Applying a proposal is explicit and only updates public behavior
controls; it does not fine-tune models, mutate hidden prompts, edit persona
prose, change runtime profiles, or export raw media.

Phase 12 adds `BilingualPerformanceBenchV3` as the release-candidate gate while
preserving the existing v1 bilingual actor bench artifacts. The wrapper
`./scripts/eval-bilingual-actor-bench.sh` now writes v1 compatibility files and
V3 release files under `artifacts/bilingual-actor-bench/`, then exits on the
V3 result. The V3 gate checks perceived performance categories, hard blockers,
quantitative metrics, optional Phase 11 preference evidence, bilingual parity,
and the public-safe avatar-adapter contract without opening the browser or
calling models, TTS, Moondream, audio, or camera services.

The browser runtime debug lane uses `scripts/evals/debug-bilingual-browser-runtime.py`
to check both primary browser paths, port status, client config, actor state,
actor events, performance state, performance events, and client-media shape
without opening a browser or calling models, TTS, Moondream, audio, or camera
services. Live WebRTC debugging should isolate browser media first, then
Moondream, and should treat camera `permission_denied` as an operator
permission issue rather than a profile regression. The 2026-04-28 camera fix is
recorded in [14 Browser Runtime Debug Stabilization](./14_debug_stabilization.md):
fresh browser frame evidence can recover stale audio-only camera hints, but
`can_see_now` still requires a fresh frame actually used by the current answer.

The V3 runtime stabilization pass adds an internal `BrowserRuntimeSessionV3`
core for live browser reliability. It owns turn-scoped camera truth, STT turn
counts, speech generation/lookahead state, stale-output drops, and primary-path
invariants. V3 ledgers, control frames, plans, memory/persona traces,
preferences, and bench evidence are derived projections from public-safe
session transitions; they must not block WebRTC connection, camera recovery,
STT, TTS, or GUI polling. The actor surface is mounted as a compact
collapsible workbench section instead of a separate competing page block.

## Privacy And Safety Boundaries

Default public state, traces, events, and bench artifacts must not expose raw
audio, raw images, SDP, ICE candidates, secrets, credentials, hidden prompts,
full model messages, raw memory bodies, or unbounded transcript text.

Live browser state may show bounded user-visible text under `live_text`.
Persistent actor traces use IDs, counts, labels, hashes, and reason codes.

Avatar adapter work is contract-only in this tranche. Future abstract/status
avatar surfaces may consume public-safe actor events, control frames, and plan
summaries, but realistic human likeness, identity cloning, face reenactment,
raw camera frames, and raw audio are outside scope.

## Phase Docs

- [01 Dual Path Baseline](./01_dual_path_baseline.md)
- [02 Actor Event Schema v2](./02_actor_event_schema_v2.md)
- [03 Browser Actor State Parity API](./03_browser_actor_state_parity_api.md)
- [04 Conversation Floor Controller](./04_conversation_floor_controller.md)
- [05 Adaptive Interruption and Echo Health](./05_adaptive_interruption_echo_health.md)
- [06 Dual TTS Speech Performance Director](./06_dual_tts_speech_performance_director.md)
- [07 Active Listener v2](./07_active_listener_v2.md)
- [08 Camera Scene State and Moondream Parity](./08_camera_scene_state_moondream_parity.md)
- [09 Persona Reference Bank and Performance Compiler v2](./09_persona_reference_bank_performance_compiler_v2.md)
- [10 Memory Continuity and Cross-Language Model](./10_memory_continuity_cross_language.md)
- [11 Browser Actor Surface v2](./11_browser_actor_surface.md)
- [12 Bilingual Performance Bench V3 and Release Gate](./12_bilingual_actor_bench_release_gate.md)
- [13 Performance Plan v3 and ActorControl Compiler](./13_performance_plan_v3.md)
- [14 Browser Runtime Debug Stabilization](./14_debug_stabilization.md)

## Validation

Use the focused release gate after actor runtime changes:

```bash
./scripts/eval-bilingual-actor-bench.sh
```

Use the broader browser workflow lane when changing launcher/runtime behavior:

```bash
uv run --extra runner --extra webrtc pytest tests/test_local_workflows.py
```

After broad cross-layer changes, run the full suite:

```bash
uv run pytest
```

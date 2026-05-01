# Engineering Review: 2026-04 Checkpoint

This document captures a full stabilization review of the Blink repository at a
meaningful project checkpoint after the local-first product path and final
Blink rename work.

It is intended to give future contributors a reliable snapshot of what was
inspected, what was fixed, what still looks risky, and what should happen next.

## Scope Reviewed

The review focused on the active local-first product surface and the repo areas
most likely to affect near-term maintainability:

- local CLI workflows under `src/blink/cli/`
- browser/WebRTC serving and the repo-owned `/client` bundle
- runtime identity, cache, and project metadata
- local scripts under `scripts/`
- the core contributor-facing docs:
  - `README.md`
  - `AGENTS.md`
  - `LOCAL_DEVELOPMENT.md`
  - `tutorial.md`
  - `docs/README.md`
  - `docs/USER_MANUAL.md`
  - `docs/chinese-conversation-adaptation.md`
  - `docs/roadmap.md`
- test and lint baselines

Verification run during this review:

- `uv run ruff check`
- `uv run python -m pytest -q`
- `uv run python -m pyright src/blink/cli/local_prefetch.py src/blink/cli/local_doctor.py src/blink/cli/local_browser.py src/blink/runner/run.py`

Observed baseline at review time:

- `931 passed, 113 skipped`
- Ruff clean
- targeted pyright is not yet a passing gate; after small safe cleanups in this
  pass, it still reports 54 issues concentrated in the local browser and runner
  layer

## Fixes Made In This Pass

### 1. Browser first-reply interruption hardening

The browser voice path had regressed relative to the protected native voice
path. The first spoken reply could be interrupted too easily even though later
turns were fine.

Fix:

- the browser runtime now applies server-side protected playback while the bot is
  speaking, so speaker bleed cannot trigger false interruptions that cut TTS off
- this does not patch `getUserMedia`, mutate browser media tracks, or rely on
  WebRTC renegotiation; `--allow-barge-in` remains the explicit echo-safe test path

Relevant files:

- `src/blink/cli/local_browser.py`
- `src/blink/cli/local_common.py`
- `tests/test_local_chat.py`
- `tests/test_local_workflows.py`

### 2. Local prefetch now respects browser-vision environment state

The prefetch CLI did not honor `BLINK_LOCAL_BROWSER_VISION` from the
environment. That made prefetch behavior inconsistent with the browser and
doctor commands.

Fix:

- `blink-local-prefetch` now enables vision prefetch when
  `BLINK_LOCAL_BROWSER_VISION` is truthy, even without `--with-vision`

Relevant files:

- `src/blink/cli/local_prefetch.py`
- `tests/test_local_prefetch.py`

### 3. Launcher script cleanup

Several local shell wrappers still carried rename fallout:

- duplicated Blink env expansions such as
  `${BLINK_LOCAL_TTS_BACKEND:-${BLINK_LOCAL_TTS_BACKEND:-}}`
- no-op legacy import-banner exports even though the runtime no longer reads
  that variable
- help text that still implied removed legacy env compatibility

Fix:

- simplified local env reads in the wrappers
- removed dead legacy import-banner exports
- removed stale compatibility messaging from bootstrap usage text

Relevant files:

- `scripts/run-local-browser.sh`
- `scripts/run-local-voice.sh`
- `scripts/run-local-chat.sh`
- `scripts/eval-local-tts.sh`
- `scripts/prefetch-local-assets.sh`
- `scripts/doctor-local-mac.sh`
- `scripts/run-local-cosyvoice-adapter.sh`
- `scripts/bootstrap-local-mac.sh`

### 4. Runtime and maintainer docs aligned with actual current behavior

The repo had several lingering contradictions after the final rename:

- docs claiming old aliases still worked
- incorrect paths to the Chinese-conversation adaptation record
- stale “legacy BLINK” phrasing
- code comments still describing the browser UI as externally owned

Fix:

- corrected current-path and current-surface documentation
- refreshed `docs/README.md` into a real documentation index
- cleaned stale code comments and local doctor guidance

Relevant files:

- `AGENTS.md`
- `LOCAL_DEVELOPMENT.md`
- `CHINESE_TTS_UPGRADE.md`
- `docs/README.md`
- `docs/roadmap.md`
- `src/blink/cli/local_browser.py`
- `src/blink/cli/local_doctor.py`
- `src/blink/runner/run.py`

### 5. Fixed a missed browser-doctor code path and reduced local typing noise

The review surfaced one concrete bug that the test suite had not been asserting
directly:

- `src/blink/cli/local_doctor.py` used `Path.cwd()` without importing `Path`
  in the browser module-check path

This pass fixed that and added a regression test for the browser profile module
check. It also tightened a few low-risk config-resolution and typing paths in
the local browser, doctor, and prefetch CLIs, which reduced the targeted
pyright backlog without changing runtime behavior.

## Bugs Found

### Fixed

1. Browser responses were too easy to interrupt.
   This was a real workflow fragility in the live browser path, especially on
   speaker-output setups. It is now protected by server-side bot-speech turn
   muting that does not alter browser media tracks.

2. `blink-local-prefetch` ignored `BLINK_LOCAL_BROWSER_VISION`.
   This created inconsistent local setup behavior and could leave the vision
   model missing even when the developer had already declared a vision-enabled
   browser workflow in `.env`.

3. Local launcher scripts still carried dead rename-era behavior.
   These were not fatal runtime bugs, but they made the active product surface
   harder to reason about and could mislead debugging.

### Not Fixed Here

1. A small number of repo-local legacy wrapper script names still exist.
   The packaged CLI surface is Blink-only, but some convenience wrappers with
   older local naming remain in the repository. Removing them is a deliberate
   breaking cleanup and should happen only if the team wants that repo-local
   boundary tightened further.

2. The optional provider layer still contains scattered backward-compatibility
   branches and technical-contract literals.
   That is expected for this framework, but it means some modules remain harder
   to simplify than the local-first path.

3. Browser camera quality still depends heavily on capture conditions and the
   Moondream path.
   The camera transport can be healthy while the answer quality remains weak.
   That is a product-quality limitation, not a transport-availability bug.

## Architecture Assessment

### Strengths

- The frame pipeline remains the right central abstraction for the repo.
- Local text, native voice, and browser voice all share a coherent runtime
  model instead of drifting into separate application stacks.
- The local-first Chinese workflow is now explicit rather than hidden in
  scattered scripts and ad hoc environment notes.
- The repo-owned browser bundle is a materially better ownership model than the
  earlier external-prebuilt branding surface.

### Current Architectural Pressure Points

1. The local-first product path is much clearer than the wider provider surface.
   The `src/blink/services/` tree remains broad and useful, but it increases
   cognitive load for contributors who only need the supported local workflow.

2. Documentation responsibility is spread across several high-value files.
   `AGENTS.md`, `LOCAL_DEVELOPMENT.md`, `tutorial.md`, `docs/USER_MANUAL.md`,
   and the Chinese-local docs all matter. That is workable, but only if the
   team keeps them synchronized aggressively after workflow changes.

3. The browser path is still the most operationally fragile local workflow.
   It depends on:
   - healthy Ollama
   - healthy TTS sidecar state when using `local-http-wav`
   - WebRTC session bootstrap
   - browser permissions
   - optional vision extras when camera analysis is desired

4. The repo still carries a narrow boundary of old-brand technical literals.
   Those are technical debt rather than branding and deserve a future cleanup
   plan if the team wants a completely Blink-native internal identity.

## Documentation Improvements Made

- Replaced the old `docs/README.md` handoff page with a real documentation index.
- Corrected broken or stale references to the Chinese adaptation record.
- Removed inaccurate “legacy aliases still work” guidance from active
  contributor docs.
- Cleaned misleading rename-era wording from the local doctor and shell
  wrappers.
- Preserved this review as repo memory so the next contributor does not have to
  rediscover the same state manually.

## Remaining Risks

These items should stay visible even though they were not directly changed here:

1. Browser voice remains the most likely user-facing failure surface.
   When failures happen, they can originate from WebRTC bootstrap, browser
   permission state, vision extras, or sidecar health rather than from the core
   pipeline.

2. The local Chinese quality target still exceeds the Kokoro bootstrap path.
   The runtime policy is correct, but high-quality Mandarin still depends on a
   healthy `local-http-wav` sidecar such as the repo-owned Melo reference path.

3. Provider modules and tracing/provider metadata can retain legacy literals
   for compatibility or external contracts.
   That remains a maintenance burden.

4. The repo can accumulate stale docs quickly because the local workflow is
   represented in both code and wrapper scripts.
   Future local CLI changes should always update:
   - `AGENTS.md`
   - `LOCAL_DEVELOPMENT.md`
   - `tutorial.md`
   - `docs/USER_MANUAL.md`
   - `env.local.example`

5. Type checking is not yet a reliable quality gate for the local runner and
   CLI surface.
   The targeted pyright run surfaced broad pre-existing issues around:
   - optional-import fallback patterns
   - loosely typed config dictionaries
   - runner route typing
   - local CLI argument resolution

## Recommended Next Priorities

1. Add a small maintained maintainer-facing debug guide for the browser path.
   Focus on:
   - permission failures
   - WebRTC bootstrap failures
   - no-audio cases
   - no-camera cases
   - sidecar health checks

2. Decide whether repo-local legacy wrapper names should be removed entirely.
   If the answer is yes, do it as an explicit cleanup change with clear release
   notes instead of leaving the repo in an ambiguous middle state.

3. Continue consolidating local workflow docs around one canonical source of
   truth for commands and one canonical source of truth for architecture.

4. Keep regression tests around the local browser path strong.
   The highest-value areas are:
   - first-turn speech reliability
   - `/client` serving and branding
   - camera/vision session behavior
   - TTS backend auto-selection policy

5. Eventually centralize or document any remaining technical-contract
   old-brand literals in one maintainable place.
   They are easier to reason about than before, but can still spread across
   provider, tracing, and protocol boundaries.

## 2026-04-17 Addendum

### Launcher ownership and shutdown cleanup

The canonical local browser path had an operational hygiene bug after the
Chinese-local hardening work: the sidecar launchers started Melo/CosyVoice in
the background and then `exec`-replaced themselves with the Blink runtime.
That dropped the cleanup trap that was supposed to own the sidecar lifecycle.

Practical effect:

- old `multiprocessing.resource_tracker` helpers and sidecar trees could remain
  after repeated stop/start cycles
- the process list became noisier over time even when the product path still
  worked

Fix:

- added shared launcher helpers in `scripts/lib/local_launcher_helpers.sh`
- updated the Melo/CosyVoice browser and voice wrappers to keep ownership of
  the Blink child process until it exits
- stopping the wrapper with `Ctrl+C` now tears down the Blink child plus any
  sidecar it spawned itself
- if the wrapper reuses an already-running sidecar, it leaves that pre-existing
  sidecar alone

Focused verification:

- `uv run python -m pytest tests/test_local_launcher_helpers.py tests/test_local_workflows.py -q`
- `bash -n` on the touched launcher scripts
- manual stop/start cycle on `./scripts/run-blink-browser.sh`

Observed result:

- stale helper trees from earlier runs were cleaned up
- a fresh browser stop/start cycle no longer accumulated orphan
  `resource_tracker` helpers
- Blink still came back healthy on the canonical Chinese Melo-backed browser
  path

### Current residual warnings after cleanup

The main local product path is working, but these warnings still remain
meaningful background debt:

1. SmallWebRTC can still emit one-time microphone or camera stall warnings
   during reconnects or browser media glitches. These are now rate-limited and
   much less noisy, but they still indicate real media instability when they
   occur.

2. The repo-owned Melo reference environment still emits third-party startup
   warnings such as `pkg_resources`, `weight_norm`, and `resume_download`.
   These are dependency hygiene issues in the isolated reference stack, not
   Blink pipeline failures.

3. The local runtime still logs:
   - `ttfs_p99_latency not set, using default 1.0s`
   - the warning about both `system_instruction` and an initial system message
     being present

4. The broader repo still carries a Python 3.13-facing `audioop` deprecation in
   `src/blink/audio/utils.py`.

These are not current blockers on the equal primary browser paths, but they
should stay on the maintenance radar.

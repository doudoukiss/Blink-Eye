# Blink Documentation

This directory is the maintained documentation surface for the current Blink
source tree. It focuses on the local-first runtime, bilingual browser product
paths, frame-pipeline architecture, memory/continuity layer, bounded embodied
actions, and the proof lanes used to keep those pieces inspectable.

## Start Here

- [`../README.md`](../README.md): project overview and quick start.
- [`../LOCAL_DEVELOPMENT.md`](../LOCAL_DEVELOPMENT.md): local setup and run
  commands.
- [`USER_MANUAL.md`](./USER_MANUAL.md): workflow guide for using Blink locally.
- [`../tutorial.md`](../tutorial.md): architecture tutorial and mental model.
- [`FILE_MAP.md`](./FILE_MAP.md): current source layout.
- [`roadmap.md`](./roadmap.md): product and runtime direction.
- [`research_references.md`](./research_references.md): GenJen paper references
  and how Blink translates them.

## Local Product Paths

- [`chinese-conversation-adaptation.md`](./chinese-conversation-adaptation.md):
  authoritative record of the Chinese-local adaptation effort.
- [`local_browser_melo.md`](./local_browser_melo.md): Chinese browser path with
  MeloTTS behind `local-http-wav`.
- [`debugging.md`](./debugging.md): local browser and voice debugging guide.
- [`debugging/native_voice_isolation.md`](./debugging/native_voice_isolation.md):
  native English voice isolation notes.
- [`native-macos-camera-helper.md`](./native-macos-camera-helper.md): separate
  macOS helper path for native camera permission.
- [`../LOCAL_CAPABILITY_MATRIX.md`](../LOCAL_CAPABILITY_MATRIX.md): what runs
  locally and what still needs hosted infrastructure.

## Runtime Architecture

- [`08_repo_integration_map.md`](./08_repo_integration_map.md): high-level
  integration map.
- [`research_references.md`](./research_references.md): source paper themes
  behind the actor runtime and performance-planning direction.
- [`bilingual_actor_runtime/README.md`](./bilingual_actor_runtime/README.md):
  actor-state, actor-events, browser UI, and bilingual release gate.
- [`performance_intelligence_v3/00_current_project_snapshot.md`](./performance_intelligence_v3/00_current_project_snapshot.md):
  current performance-intelligence baseline.
- [`CONTINUITY_TEST_MANUAL.md`](./CONTINUITY_TEST_MANUAL.md): manual continuity
  validation.
- [`WORKING_MEMORY_QA.md`](./WORKING_MEMORY_QA.md),
  [`PROCEDURAL_MEMORY_QA.md`](./PROCEDURAL_MEMORY_QA.md), and
  [`TEMPORAL_CONTINUITY_GRAPH_QA.md`](./TEMPORAL_CONTINUITY_GRAPH_QA.md):
  memory and continuity QA surfaces.

## Safety, Privacy, And QA

- [`06_safety_privacy_consent.md`](./06_safety_privacy_consent.md): browser
  runtime safety and consent boundaries.
- [`MANUAL_QA_SCORECARD.md`](./MANUAL_QA_SCORECARD.md): human-facing QA rubric.
- [`FUZZ_TARGET_MATRIX.md`](./FUZZ_TARGET_MATRIX.md) and
  [`FUZZ_TEST_QUICKSTART.md`](./FUZZ_TEST_QUICKSTART.md): fuzz/proof lane
  orientation.
- [`ENGINEERING_REVIEW_2026-04.md`](./ENGINEERING_REVIEW_2026-04.md):
  engineering review snapshot.

## Embodiment And Robot Head

- [`ROBOT_HEAD_INTEGRATION.md`](./ROBOT_HEAD_INTEGRATION.md)
- [`ROBOT_HEAD_OPERATOR_HANDBOOK.md`](./ROBOT_HEAD_OPERATOR_HANDBOOK.md)
- [`ROBOT_HEAD_CAPABILITY_CATALOG.md`](./ROBOT_HEAD_CAPABILITY_CATALOG.md)
- [`robot_head_hardware_and_serial_handoff.md`](./robot_head_hardware_and_serial_handoff.md)
- [`robot_head_live_limits.md`](./robot_head_live_limits.md)

## Documentation Status

This repository is now kept source-first. Generated browser package copies,
runtime logs, local build products, virtual environments, and internal planning
bundles are intentionally absent from the public tree. Required runtime assets
that Blink loads directly, including the vendored browser client assets under
`web/client_src/src`, are retained. When a doc mentions generated output, treat
it as a local rebuild target, not committed source.

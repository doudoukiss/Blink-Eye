#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

lane="all"

usage() {
  cat <<'EOF'
Usage: ./scripts/test-brain-core.sh [--lane fast|proof|fuzz-smoke|atheris|all] [--list-lanes]

Canonical headless proof entrypoint for Blink's brain core.

Lanes:
  fast        Import hygiene, replay, packet proofs, continuity evals, audit reports,
              planning, continuity memory, and active-state unit coverage.
  proof       tests/brain_properties plus tests/brain_stateful.
  fuzz-smoke  Deterministic tests/test_brain_fuzz_harness_smoke.py.
  atheris     Opt-in coverage-guided harnesses for libFuzzer-capable machines.
  all         fast + proof + fuzz-smoke.
EOF
}

list_lanes() {
  printf '%s\n' fast proof fuzz-smoke atheris all
}

run_uv_pytest() {
  printf '==> uv run pytest %s -q\n' "$*"
  (cd "${REPO_ROOT}" && uv run pytest "$@" -q)
}

run_atheris_target() {
  local target="$1"
  local runs="${ATHERIS_RUNS:-200}"
  printf '==> uv run --with atheris python %s -atheris_runs=%s\n' "${target}" "${runs}"
  (cd "${REPO_ROOT}" && uv run --with atheris python "${target}" "-atheris_runs=${runs}")
}

run_fast_lane() {
  run_uv_pytest \
    tests/test_brain_import_hygiene.py \
    tests/test_brain_adapters.py \
    tests/brain_core/test_replay.py \
    tests/test_brain_replay.py \
    tests/test_brain_world_model.py \
    tests/test_brain_context_packet_proofs.py \
    tests/test_brain_context_policy.py \
    tests/test_brain_continuity_evals.py \
    tests/test_brain_audit_reports.py \
    tests/test_brain_adapter_promotion.py \
    tests/test_brain_episode_export.py \
    tests/test_brain_failure_clusters.py \
    tests/test_brain_practice_director.py \
    tests/test_brain_runtime_shell_contract.py \
    tests/test_brain_sim_to_real_report.py \
    tests/test_brain_skill_evidence.py \
    tests/test_brain_commitments.py \
    tests/test_brain_planning.py \
    tests/test_brain_rehearsal.py \
    tests/test_brain_embodied_executive.py \
    tests/test_brain_memory_v2.py \
    tests/test_brain_scene_world_state.py \
    tests/test_brain_private_working_memory.py \
    tests/test_brain_active_situation_model.py
}

run_proof_lane() {
  run_uv_pytest tests/brain_properties
  run_uv_pytest tests/brain_stateful
}

run_fuzz_smoke_lane() {
  run_uv_pytest tests/test_brain_fuzz_harness_smoke.py
}

run_atheris_lane() {
  run_atheris_target tests/fuzz/atheris/fuzz_graph_projection.py
  run_atheris_target tests/fuzz/atheris/fuzz_replay_artifacts.py
  run_atheris_target tests/fuzz/atheris/fuzz_active_state_projections.py
  run_atheris_target tests/fuzz/atheris/fuzz_counterfactual_rehearsal.py
  run_atheris_target tests/fuzz/atheris/fuzz_embodied_executive.py
  run_atheris_target tests/fuzz/atheris/fuzz_reevaluation_digests.py
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lane)
      if [[ $# -lt 2 ]]; then
        usage >&2
        exit 1
      fi
      lane="$2"
      shift 2
      ;;
    --list-lanes)
      list_lanes
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${lane}" in
  fast)
    run_fast_lane
    ;;
  proof)
    run_proof_lane
    ;;
  fuzz-smoke)
    run_fuzz_smoke_lane
    ;;
  atheris)
    run_atheris_lane
    ;;
  all)
    run_fast_lane
    run_proof_lane
    run_fuzz_smoke_lane
    ;;
  *)
    printf 'Unknown lane: %s\n' "${lane}" >&2
    usage >&2
    exit 1
    ;;
esac

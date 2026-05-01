#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

lane="all"

usage() {
  cat <<'EOF'
Usage: ./scripts/test-embodied-core.sh [--lane fast|proof|all] [--list-lanes]

Canonical proof entrypoint for Blink's embodied and perception slice.

Lanes:
  fast   Perception broker, embodied shell-contract slices, robot-head policy/runtime,
         capability dispatch, and finite embodied action coverage.
  proof  Scene-world and multimodal autobiography properties plus embodied-adjacent
         stateful coverage.
  all    fast + proof.
EOF
}

list_lanes() {
  printf '%s\n' fast proof all
}

run_uv_pytest() {
  printf '==> uv run pytest %s -q\n' "$*"
  (cd "${REPO_ROOT}" && uv run pytest "$@" -q)
}

run_fast_lane() {
  run_uv_pytest \
    tests/test_brain_adapters.py \
    tests/test_brain_perception_broker.py \
    tests/test_brain_runtime_shell_contract.py \
    tests/test_robot_head_policy.py \
    tests/test_robot_head_simulation_runtime.py \
    tests/test_brain_capability_registry.py \
    tests/test_brain_actions.py \
    tests/test_brain_rehearsal.py \
    tests/test_brain_embodied_executive.py
}

run_proof_lane() {
  run_uv_pytest \
    tests/brain_properties/test_scene_world_state_properties.py \
    tests/brain_properties/test_multimodal_autobiography_properties.py \
    tests/brain_properties/test_embodied_executive_properties.py \
    tests/brain_stateful/test_multimodal_memory_retention_state_machine.py \
    tests/brain_stateful/test_policy_coupled_presence_state_machine.py \
    tests/brain_stateful/test_embodied_executive_state_machine.py
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
  all)
    run_fast_lane
    run_proof_lane
    ;;
  *)
    printf 'Unknown lane: %s\n' "${lane}" >&2
    usage >&2
    exit 1
    ;;
esac

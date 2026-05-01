#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

lane="all"

usage() {
  cat <<'EOF'
Usage: ./scripts/test-embodied-evals.sh [--lane smoke|benchmark|all] [--list-lanes]

Canonical Phase 21A embodied eval entrypoint for Blink.

Lanes:
  smoke      Deterministic CI-suitable embodied eval suite.
  benchmark  Deterministic benchmark-suite skeleton for targeted or scheduled runs.
  all        smoke + benchmark.
EOF
}

list_lanes() {
  printf '%s\n' smoke benchmark all
}

run_suite() {
  local suite="$1"
  printf '==> uv run blink-embodied-eval --suite %s\n' "${suite}"
  (cd "${REPO_ROOT}" && uv run blink-embodied-eval --suite "${suite}")
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
  smoke)
    run_suite smoke
    ;;
  benchmark)
    run_suite benchmark
    ;;
  all)
    run_suite smoke
    run_suite benchmark
    ;;
  *)
    printf 'Unknown lane: %s\n' "${lane}" >&2
    usage >&2
    exit 1
    ;;
esac

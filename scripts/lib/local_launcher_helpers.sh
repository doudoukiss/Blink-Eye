#!/usr/bin/env bash

load_env_defaults() {
  local env_file="${1:-.env}"
  if [[ ! -f "$env_file" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" != *=* ]] && continue

    local key="${line%%=*}"
    local value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ -z "$key" ]] && continue
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    if [[ -n "${!key+x}" ]]; then
      continue
    fi

    value="${value%$'\r'}"
    if [[ "$value" =~ ^\".*\"$ ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" =~ ^\'.*\'$ ]]; then
      value="${value:1:${#value}-2}"
    fi

    export "$key=$value"
  done < "$env_file"
}

wait_for_http() {
  local url="$1"
  local timeout_secs="${2:-120}"
  local start_ts
  start_ts="$(date +%s)"

  while true; do
    if curl --silent --fail "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= timeout_secs )); then
      echo "Timed out waiting for $url" >&2
      return 1
    fi
    sleep 2
  done
}

terminate_pid_tree() {
  local pid="$1"
  local label="${2:-process}"
  local grace_secs="${3:-5}"
  local child_pids=""
  local deadline=0

  if [[ -z "$pid" ]] || ! kill -0 "$pid" >/dev/null 2>&1; then
    return 0
  fi

  child_pids="$(pgrep -P "$pid" || true)"
  kill "$pid" >/dev/null 2>&1 || true
  deadline=$((SECONDS + grace_secs))

  while kill -0 "$pid" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "Force-stopping ${label} (pid ${pid}) after graceful shutdown timed out." >&2
      kill -9 "$pid" >/dev/null 2>&1 || true
      break
    fi
    sleep 0.2
  done

  for child_pid in $child_pids; do
    if kill -0 "$child_pid" >/dev/null 2>&1; then
      kill "$child_pid" >/dev/null 2>&1 || true
      sleep 0.1
      kill -9 "$child_pid" >/dev/null 2>&1 || true
    fi
  done
}

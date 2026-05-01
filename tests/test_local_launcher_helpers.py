import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "scripts" / "lib" / "local_launcher_helpers.sh"


def _run_helper_script(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_wait_for_http_returns_when_endpoint_is_ready():
    script = f"""
set -euo pipefail
source "{HELPERS}"
python -m http.server 19321 >/dev/null 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" >/dev/null 2>&1 || true' EXIT
wait_for_http "http://127.0.0.1:19321" 5
"""

    result = _run_helper_script(script)
    assert result.returncode == 0


def test_terminate_pid_tree_stops_parent_and_child():
    script = f"""
set -euo pipefail
source "{HELPERS}"
bash -c 'sleep 30 & wait' &
PARENT_PID=$!
sleep 0.5
CHILD_PID="$(pgrep -P "$PARENT_PID")"
terminate_pid_tree "$PARENT_PID" "test launcher" 1
sleep 0.5
if kill -0 "$PARENT_PID" >/dev/null 2>&1; then
  echo "parent still alive" >&2
  exit 1
fi
if kill -0 "$CHILD_PID" >/dev/null 2>&1; then
  echo "child still alive" >&2
  exit 1
fi
"""

    result = _run_helper_script(script)
    assert result.returncode == 0

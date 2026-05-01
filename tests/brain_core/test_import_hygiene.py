import json
import os
from pathlib import Path
import subprocess
import sys


def test_brain_core_import_does_not_load_provider_or_transport_modules():
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                "import blink.brain.core; "
                "tracked = {"
                "'blink.services': 'blink.services' in sys.modules, "
                "'blink.transports': 'blink.transports' in sys.modules, "
                "'openai': 'openai' in sys.modules"
                "}; "
                "print(json.dumps(tracked, sort_keys=True))"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = json.loads(process.stdout.strip().splitlines()[-1])
    assert tracked == {
        "blink.services": False,
        "blink.transports": False,
        "openai": False,
    }

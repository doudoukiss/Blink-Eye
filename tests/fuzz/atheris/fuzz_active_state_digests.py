"""Atheris harness for active-state digest builders.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_active_state_digests.py -atheris_runs=1000
"""

from __future__ import annotations

import json
import sys

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.active_situation_model_digest import build_active_situation_model_digest
    from blink.brain.private_working_memory_digest import build_private_working_memory_digest
    from blink.brain.scene_world_state_digest import build_scene_world_state_digest


def _assert_json_safe(value) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)


def TestOneInput(data: bytes) -> None:
    root = structured_inputs.active_state_projection_mapping(
        structured_inputs.decode_jsonish_input(data)
    )

    private_digest = build_private_working_memory_digest(
        private_working_memory=root.get("private_working_memory"),
    )
    _assert_json_safe(private_digest)

    active_digest = build_active_situation_model_digest(
        active_situation_model=root.get("active_situation_model"),
    )
    _assert_json_safe(active_digest)

    scene_digest = build_scene_world_state_digest(
        scene_world_state=root.get("scene_world_state"),
    )
    _assert_json_safe(scene_digest)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

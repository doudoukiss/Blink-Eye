"""Atheris harness for active-state projection parsing.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_active_state_projections.py -atheris_runs=1000
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.projections import (
        BrainActiveSituationProjection,
        BrainActiveSituationRecord,
        BrainPrivateWorkingMemoryProjection,
        BrainPrivateWorkingMemoryRecord,
        BrainSceneWorldAffordanceRecord,
        BrainSceneWorldEntityRecord,
        BrainSceneWorldProjection,
    )


def _exercise_roundtrip(
    parser: Callable[[dict[str, Any] | None], Any],
    payload: dict[str, Any],
) -> None:
    try:
        record = parser(payload)
    except structured_inputs.SAFE_REJECTION_EXCEPTIONS:
        return
    if record is None:
        return
    roundtrip = parser(record.as_dict())
    assert roundtrip is not None
    assert roundtrip.as_dict() == record.as_dict()


def TestOneInput(data: bytes) -> None:
    root = structured_inputs.active_state_projection_mapping(
        structured_inputs.decode_jsonish_input(data)
    )

    pwm_payload = structured_inputs.as_mapping(
        root.get("private_working_memory"),
        field_name="private_working_memory",
    )
    for record_payload in structured_inputs.as_list(pwm_payload.get("records")):
        _exercise_roundtrip(
            BrainPrivateWorkingMemoryRecord.from_dict,
            structured_inputs.as_mapping(record_payload, field_name="record"),
        )
    _exercise_roundtrip(BrainPrivateWorkingMemoryProjection.from_dict, pwm_payload)

    situation_payload = structured_inputs.as_mapping(
        root.get("active_situation_model"),
        field_name="active_situation_model",
    )
    for record_payload in structured_inputs.as_list(situation_payload.get("records")):
        _exercise_roundtrip(
            BrainActiveSituationRecord.from_dict,
            structured_inputs.as_mapping(record_payload, field_name="record"),
        )
    _exercise_roundtrip(BrainActiveSituationProjection.from_dict, situation_payload)

    scene_payload = structured_inputs.as_mapping(root.get("scene_world_state"), field_name="scene_world_state")
    for entity_payload in structured_inputs.as_list(scene_payload.get("entities")):
        _exercise_roundtrip(
            BrainSceneWorldEntityRecord.from_dict,
            structured_inputs.as_mapping(entity_payload, field_name="entity"),
        )
    for affordance_payload in structured_inputs.as_list(scene_payload.get("affordances")):
        _exercise_roundtrip(
            BrainSceneWorldAffordanceRecord.from_dict,
            structured_inputs.as_mapping(affordance_payload, field_name="affordance"),
        )
    _exercise_roundtrip(BrainSceneWorldProjection.from_dict, scene_payload)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

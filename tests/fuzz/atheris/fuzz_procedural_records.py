"""Atheris harness for procedural and planning record parsing.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_procedural_records.py -atheris_runs=1000
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.memory_v2.procedural import (
        BrainProceduralExecutionTraceRecord,
        BrainProceduralOutcomeRecord,
        BrainProceduralStepTraceRecord,
        BrainProceduralTraceProjection,
    )
    from blink.brain.memory_v2.skills import (
        BrainProceduralActivationConditionRecord,
        BrainProceduralEffectRecord,
        BrainProceduralFailureSignatureRecord,
        BrainProceduralInvariantRecord,
        BrainProceduralSkillProjection,
        BrainProceduralSkillRecord,
        BrainProceduralSkillStatsRecord,
    )
    from blink.brain.procedural_planning import (
        BrainPlanningSkillCandidate,
        BrainPlanningSkillDelta,
        BrainPlanningSkillRejection,
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
    value = structured_inputs.decode_jsonish_input(data)
    root = structured_inputs.as_mapping(value, field_name="procedural")

    parser_specs: tuple[tuple[Callable[[dict[str, Any] | None], Any], str], ...] = (
        (BrainProceduralStepTraceRecord.from_dict, "step"),
        (BrainProceduralExecutionTraceRecord.from_dict, "trace"),
        (BrainProceduralOutcomeRecord.from_dict, "outcome"),
        (BrainProceduralTraceProjection.from_dict, "projection"),
        (BrainProceduralActivationConditionRecord.from_dict, "condition"),
        (BrainProceduralInvariantRecord.from_dict, "invariant"),
        (BrainProceduralEffectRecord.from_dict, "effect"),
        (BrainProceduralFailureSignatureRecord.from_dict, "failure_signature"),
        (BrainProceduralSkillStatsRecord.from_dict, "stats"),
        (BrainProceduralSkillRecord.from_dict, "skill"),
        (BrainProceduralSkillProjection.from_dict, "skill_projection"),
        (BrainPlanningSkillRejection.from_dict, "rejection"),
        (BrainPlanningSkillDelta.from_dict, "delta"),
        (BrainPlanningSkillCandidate.from_dict, "candidate"),
    )

    for parser, key in parser_specs:
        _exercise_roundtrip(parser, root)
        if key in root:
            _exercise_roundtrip(parser, structured_inputs.as_mapping(root.get(key), field_name=key))


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

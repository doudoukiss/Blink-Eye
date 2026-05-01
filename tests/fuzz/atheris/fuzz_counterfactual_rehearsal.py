"""Atheris harness for counterfactual rehearsal records and digests.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_counterfactual_rehearsal.py -atheris_runs=1000
"""

from __future__ import annotations

import json
import sys

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.counterfactual_rehearsal_digest import build_counterfactual_rehearsal_digest
    from blink.brain.counterfactuals import (
        append_outcome_comparison,
        append_rehearsal_request,
        append_rehearsal_result,
    )
    from blink.brain.projections import (
        BrainActionOutcomeComparisonRecord,
        BrainActionRehearsalRequest,
        BrainActionRehearsalResult,
        BrainCounterfactualRehearsalProjection,
    )


def _assert_json_safe(value) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)


def TestOneInput(data: bytes) -> None:
    root = structured_inputs.decode_jsonish_input(data)
    mapping = structured_inputs.as_mapping(root, field_name="counterfactual_root")
    projection_payload = structured_inputs.as_mapping(
        mapping.get("counterfactual_rehearsal", mapping),
        field_name="counterfactual_rehearsal",
    )
    projection = BrainCounterfactualRehearsalProjection.from_dict(
        {
            "scope_key": str(projection_payload.get("scope_key", "thread-fuzz")),
            "presence_scope_key": str(
                projection_payload.get("presence_scope_key", "browser:presence")
            ),
            "open_requests": projection_payload.get("open_requests", []),
            "recent_rehearsals": projection_payload.get("recent_rehearsals", []),
            "recent_comparisons": projection_payload.get("recent_comparisons", []),
            "calibration_summary": projection_payload.get("calibration_summary", {}),
            "updated_at": str(
                projection_payload.get("updated_at", "2026-01-01T00:00:00+00:00")
            ),
        }
    )
    projection.sync_lists()
    _assert_json_safe(projection.as_dict())
    _assert_json_safe(
        build_counterfactual_rehearsal_digest(counterfactual_rehearsal=projection.as_dict())
    )

    request = BrainActionRehearsalRequest.from_dict(
        structured_inputs.as_mapping(mapping.get("request", {}), field_name="request")
    )
    result = BrainActionRehearsalResult.from_dict(
        structured_inputs.as_mapping(mapping.get("result", {}), field_name="result")
    )
    comparison = BrainActionOutcomeComparisonRecord.from_dict(
        structured_inputs.as_mapping(mapping.get("comparison", {}), field_name="comparison")
    )

    if request is not None:
        append_rehearsal_request(projection, request)
    if result is not None:
        append_rehearsal_result(projection, result)
    if comparison is not None:
        append_outcome_comparison(projection, comparison)

    _assert_json_safe(projection.as_dict())
    _assert_json_safe(
        build_counterfactual_rehearsal_digest(counterfactual_rehearsal=projection.as_dict())
    )


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

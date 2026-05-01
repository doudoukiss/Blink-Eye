"""Atheris harness for embodied executive records and digests.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_embodied_executive.py -atheris_runs=1000
"""

from __future__ import annotations

import json
import sys

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.embodied_executive import (
        append_embodied_action_envelope,
        append_embodied_execution_trace,
        append_embodied_intent,
        append_embodied_recovery,
    )
    from blink.brain.embodied_executive_digest import build_embodied_executive_digest
    from blink.brain.projections import (
        BrainEmbodiedActionEnvelope,
        BrainEmbodiedExecutionTrace,
        BrainEmbodiedExecutiveProjection,
        BrainEmbodiedIntent,
        BrainEmbodiedRecoveryRecord,
    )


def _assert_json_safe(value) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)


def TestOneInput(data: bytes) -> None:
    root = structured_inputs.decode_jsonish_input(data)
    mapping = structured_inputs.as_mapping(root, field_name="embodied_root")
    projection_payload = structured_inputs.as_mapping(
        mapping.get("embodied_executive", mapping),
        field_name="embodied_executive",
    )
    projection = BrainEmbodiedExecutiveProjection.from_dict(
        {
            "scope_key": str(projection_payload.get("scope_key", "thread-fuzz")),
            "presence_scope_key": str(
                projection_payload.get("presence_scope_key", "browser:presence")
            ),
            "current_intent": projection_payload.get("current_intent"),
            "recent_action_envelopes": projection_payload.get("recent_action_envelopes", []),
            "recent_execution_traces": projection_payload.get("recent_execution_traces", []),
            "recent_recoveries": projection_payload.get("recent_recoveries", []),
            "updated_at": str(
                projection_payload.get("updated_at", "2026-01-01T00:00:00+00:00")
            ),
        }
    )
    projection.sync_lists()
    _assert_json_safe(projection.as_dict())
    _assert_json_safe(build_embodied_executive_digest(embodied_executive=projection.as_dict()))

    intent = BrainEmbodiedIntent.from_dict(
        structured_inputs.as_mapping(mapping.get("intent", {}), field_name="intent")
    )
    envelope = BrainEmbodiedActionEnvelope.from_dict(
        structured_inputs.as_mapping(mapping.get("envelope", {}), field_name="envelope")
    )
    trace = BrainEmbodiedExecutionTrace.from_dict(
        structured_inputs.as_mapping(mapping.get("trace", {}), field_name="trace")
    )
    recovery = BrainEmbodiedRecoveryRecord.from_dict(
        structured_inputs.as_mapping(mapping.get("recovery", {}), field_name="recovery")
    )

    if intent is not None:
        append_embodied_intent(projection, intent)
    if envelope is not None:
        append_embodied_action_envelope(projection, envelope)
    if trace is not None:
        append_embodied_execution_trace(projection, trace)
    if recovery is not None:
        append_embodied_recovery(projection, recovery)

    low_level_actions = [
        {
            "action_id": str(item.get("action_id", "cmd_look_left")),
            "source": str(item.get("source", "operator")),
            "accepted": bool(item.get("accepted", False)),
            "preview_only": bool(item.get("preview_only", False)),
            "summary": str(item.get("summary", "")),
            "created_at": str(item.get("created_at", "2026-01-01T00:00:00+00:00")),
        }
        for item in structured_inputs.as_list(mapping.get("recent_action_events", []))
        if isinstance(item, dict)
    ]

    _assert_json_safe(projection.as_dict())
    _assert_json_safe(
        build_embodied_executive_digest(
            embodied_executive=projection.as_dict(),
            recent_action_events=low_level_actions,
        )
    )


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

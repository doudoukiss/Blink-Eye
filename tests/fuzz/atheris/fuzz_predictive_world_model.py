"""Atheris harness for predictive world-model records and digests.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_predictive_world_model.py -atheris_runs=1000
"""

from __future__ import annotations

import json
import sys

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.events import BrainEventRecord
    from blink.brain.projections import (
        BrainActiveSituationProjection,
        BrainPredictionRecord,
        BrainPredictiveWorldModelProjection,
        BrainSceneWorldProjection,
    )
    from blink.brain.world_model import (
        append_prediction_generation,
        append_prediction_resolution,
        build_prediction_resolution,
        resolve_prediction_against_state,
    )
    from blink.brain.world_model_digest import build_world_model_digest


def _event_record(payload: dict, *, index: int) -> BrainEventRecord:
    return BrainEventRecord(
        id=int(payload.get("id", index + 1)),
        event_id=str(payload.get("event_id", f"evt-predictive-{index}")),
        event_type=str(payload.get("event_type", "scene.changed")),
        ts=str(payload.get("ts", f"2026-01-01T00:00:0{index}+00:00")),
        agent_id=str(payload.get("agent_id", "agent-fuzz")),
        user_id=str(payload.get("user_id", "user-fuzz")),
        session_id=str(payload.get("session_id", "session-fuzz")),
        thread_id=str(payload.get("thread_id", "thread-fuzz")),
        source=str(payload.get("source", "atheris")),
        correlation_id=payload.get("correlation_id"),
        causal_parent_id=payload.get("causal_parent_id"),
        confidence=float(payload.get("confidence", 1.0)),
        payload_json=json.dumps(payload.get("payload", {}), ensure_ascii=False),
        tags_json=json.dumps(payload.get("tags", []), ensure_ascii=False),
    )


def _assert_json_safe(value) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)


def TestOneInput(data: bytes) -> None:
    root = structured_inputs.decode_jsonish_input(data)
    mapping = structured_inputs.as_mapping(root, field_name="predictive_root")
    predictive_payload = structured_inputs.as_mapping(
        mapping.get("predictive_world_model", mapping),
        field_name="predictive_world_model",
    )
    projection = BrainPredictiveWorldModelProjection.from_dict(
        {
            "scope_key": str(predictive_payload.get("scope_key", "thread-fuzz")),
            "presence_scope_key": str(
                predictive_payload.get("presence_scope_key", "browser:presence")
            ),
            "active_predictions": predictive_payload.get("active_predictions", []),
            "recent_resolutions": predictive_payload.get("recent_resolutions", []),
            "calibration_summary": predictive_payload.get("calibration_summary", {}),
            "updated_at": str(
                predictive_payload.get("updated_at", "2026-01-01T00:00:00+00:00")
            ),
        }
    )
    projection.sync_lists()
    _assert_json_safe(projection.as_dict())
    _assert_json_safe(build_world_model_digest(predictive_world_model=projection.as_dict()))

    prediction = BrainPredictionRecord.from_dict(
        structured_inputs.as_mapping(mapping.get("prediction", {}), field_name="prediction")
    )
    active_state_inputs = structured_inputs.active_state_projection_mapping(root)
    scene_world_state = BrainSceneWorldProjection.from_dict(active_state_inputs["scene_world_state"])
    active_situation_model = BrainActiveSituationProjection.from_dict(
        active_state_inputs["active_situation_model"]
    )
    event_payloads = structured_inputs.as_list(mapping.get("events", []))
    events = [
        _event_record(
            structured_inputs.as_mapping(item, field_name="event"),
            index=index,
        )
        for index, item in enumerate(event_payloads)
    ]

    if prediction is None:
        return

    append_prediction_generation(projection, prediction)
    if events:
        try:
            resolved = resolve_prediction_against_state(
                prediction=prediction,
                trigger_event=events[0],
                scene_world_state=scene_world_state,
                active_situation_model=active_situation_model,
            )
        except structured_inputs.SAFE_REJECTION_EXCEPTIONS:
            resolved = None
        if resolved is not None:
            resolution = build_prediction_resolution(
                prediction=prediction,
                trigger_event=events[0],
                resolution_kind="confirmed" if resolved else "invalidated",
                resolution_summary="fuzz-resolution",
            )
            append_prediction_resolution(projection, resolution)
    _assert_json_safe(projection.as_dict())
    _assert_json_safe(build_world_model_digest(predictive_world_model=projection.as_dict()))


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

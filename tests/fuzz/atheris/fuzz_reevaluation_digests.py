"""Atheris harness for autonomy and reevaluation digest builders.

Run with:
    uv run --with atheris python tests/fuzz/atheris/fuzz_reevaluation_digests.py -atheris_runs=1000
"""

from __future__ import annotations

import json
import sys

import _structured_inputs as structured_inputs
import atheris

with atheris.instrument_imports():
    from blink.brain.autonomy import BrainAutonomyLedgerProjection
    from blink.brain.autonomy_digest import build_autonomy_digest
    from blink.brain.events import BrainEventRecord
    from blink.brain.projections import BrainAgendaProjection
    from blink.brain.reevaluation_digest import build_reevaluation_digest


def _json_safe(value) -> None:
    encoded = json.dumps(value, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert isinstance(decoded, dict)


def _event_record(payload: dict, *, index: int) -> BrainEventRecord:
    return BrainEventRecord(
        id=int(payload.get("id", index + 1)),
        event_id=str(payload.get("event_id", f"evt-{index}")),
        event_type=str(payload.get("event_type", "goal.candidate.created")),
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


def TestOneInput(data: bytes) -> None:
    root = structured_inputs.decode_jsonish_input(data)

    autonomy_inputs = structured_inputs.autonomy_digest_inputs(root)
    autonomy_ledger = BrainAutonomyLedgerProjection.from_dict(autonomy_inputs["autonomy_ledger"])
    agenda = BrainAgendaProjection.from_dict(autonomy_inputs["agenda"])
    autonomy_digest = build_autonomy_digest(
        autonomy_ledger=autonomy_ledger,
        agenda=agenda,
    )
    _json_safe(autonomy_digest)

    reevaluation_inputs = structured_inputs.reevaluation_digest_inputs(root)
    reevaluation_ledger = BrainAutonomyLedgerProjection.from_dict(
        reevaluation_inputs["autonomy_ledger"]
    )
    recent_events = [
        _event_record(payload, index=index)
        for index, payload in enumerate(reevaluation_inputs["recent_events"])
    ]
    reevaluation_digest = build_reevaluation_digest(
        autonomy_ledger=reevaluation_ledger,
        recent_events=recent_events,
    )
    _json_safe(reevaluation_digest)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()

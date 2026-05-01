import json

from blink.brain.capabilities import CapabilityRegistry
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import load_default_agent_blocks
from blink.brain.projections import (
    BrainBlockedReason,
    BrainBlockedReasonKind,
    BrainWakeCondition,
    BrainWakeConditionKind,
)
from blink.brain.session import resolve_brain_session_ids
from blink.brain.store import BrainStore
from blink.cli.local_brain_shell import main as local_brain_shell_main


def _seed_shell_db(tmp_path, *, deferred: bool) -> tuple[BrainStore, object, str]:
    store = BrainStore(path=tmp_path / "brain.db")
    store.ensure_default_blocks(load_default_agent_blocks())
    session_ids = resolve_brain_session_ids(runtime_kind="browser", client_id="alpha")
    executive = BrainExecutive(
        store=store,
        session_resolver=lambda: session_ids,
        capability_registry=CapabilityRegistry(),
    )
    executive.create_commitment_goal(
        title="Inspect shell CLI commitment",
        intent="narrative.commitment",
        source="test",
        details={"summary": "Need stable shell CLI coverage."},
    )
    commitment = store.list_executive_commitments(user_id=session_ids.user_id, limit=4)[0]
    store.add_episode(
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        user_text="Show me the shell state.",
        assistant_text="Showing the shell state.",
        assistant_summary="Showed the shell state.",
        tool_calls=[],
    )
    if deferred:
        executive.defer_commitment(
            commitment_id=commitment.commitment_id,
            reason=BrainBlockedReason(
                kind=BrainBlockedReasonKind.EXPLICIT_DEFER.value,
                summary="Waiting for an explicit resume.",
            ),
            wake_conditions=[
                BrainWakeCondition(
                    kind=BrainWakeConditionKind.EXPLICIT_RESUME.value,
                    summary="Resume explicitly when ready.",
                    details={},
                )
            ],
        )
    return store, session_ids, commitment.commitment_id


def test_local_brain_shell_cli_snapshot_packet_and_waits_emit_stable_json(tmp_path, capsys):
    store, session_ids, _ = _seed_shell_db(tmp_path, deferred=True)
    store.close()
    common_args = [
        "--brain-db-path",
        str(tmp_path / "brain.db"),
        "--runtime-kind",
        "browser",
        "--client-id",
        "alpha",
        "--user-id",
        session_ids.user_id,
        "--thread-id",
        session_ids.thread_id,
    ]

    assert local_brain_shell_main(["snapshot", *common_args]) == 0
    snapshot = json.loads(capsys.readouterr().out)
    assert snapshot["session_ids"]["thread_id"] == session_ids.thread_id
    assert "autonomy_digest" in snapshot

    assert local_brain_shell_main(
        ["packet", *common_args, "--task", "reply", "--query", "What is waiting?"]
    ) == 0
    packet = json.loads(capsys.readouterr().out)
    assert packet["task"] == "reply"
    assert packet["packet_digest"]

    assert local_brain_shell_main(["waits", *common_args]) == 0
    waits = json.loads(capsys.readouterr().out)
    assert waits["current_wait_kind_counts"]["explicit_resume"] >= 1
    assert waits["wake_digest"]["current_waiting_commitments"]


def test_local_brain_shell_cli_controls_append_runtime_shell_events(tmp_path, capsys):
    store, session_ids, commitment_id = _seed_shell_db(tmp_path, deferred=False)
    store.close()
    common_args = [
        "--brain-db-path",
        str(tmp_path / "brain.db"),
        "--runtime-kind",
        "browser",
        "--client-id",
        "alpha",
        "--user-id",
        session_ids.user_id,
        "--thread-id",
        session_ids.thread_id,
    ]

    assert local_brain_shell_main(
        [
            "interrupt",
            *common_args,
            "--commitment-id",
            commitment_id,
            "--reason",
            "Pause the commitment now.",
        ]
    ) == 0
    interrupted = json.loads(capsys.readouterr().out)

    assert local_brain_shell_main(
        [
            "suppress",
            *common_args,
            "--commitment-id",
            commitment_id,
            "--reason",
            "Hold until the operator resumes it.",
        ]
    ) == 0
    suppressed = json.loads(capsys.readouterr().out)

    assert local_brain_shell_main(
        [
            "resume",
            *common_args,
            "--commitment-id",
            commitment_id,
            "--reason",
            "Operator approved the resume.",
        ]
    ) == 0
    resumed = json.loads(capsys.readouterr().out)

    reopened = BrainStore(path=tmp_path / "brain.db")
    try:
        events = reopened.recent_brain_events(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            limit=12,
        )
    finally:
        reopened.close()

    assert interrupted["applied"] is True
    assert suppressed["applied"] is True
    assert resumed["applied"] is True
    assert any(
        event.event_type == BrainEventType.GOAL_DEFERRED
        and event.source == "runtime_shell"
        and dict(event.payload or {}).get("runtime_shell_control", {}).get("control_kind")
        == "interrupt"
        for event in events
    )
    assert any(
        event.event_type == BrainEventType.GOAL_DEFERRED
        and event.source == "runtime_shell"
        and dict(event.payload or {}).get("runtime_shell_control", {}).get("control_kind")
        == "suppress"
        for event in events
    )
    assert any(
        event.event_type == BrainEventType.GOAL_RESUMED
        and event.source == "runtime_shell"
        and dict(event.payload or {}).get("runtime_shell_control", {}).get("control_kind")
        == "resume"
        for event in events
    )

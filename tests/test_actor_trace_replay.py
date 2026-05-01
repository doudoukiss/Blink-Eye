import json
import os
import subprocess
import sys

from blink.interaction.actor_events import (
    ActorEventModeV2,
    ActorEventTypeV2,
    ActorEventV2,
    ActorTraceWriter,
    replay_actor_trace,
)
from blink.interaction.performance_events import BrowserInteractionMode, BrowserPerformanceEventBus


def _event_payload(
    event_id: int,
    event_type: ActorEventTypeV2,
    mode: ActorEventModeV2,
    *,
    timestamp: str,
) -> dict[str, object]:
    return ActorEventV2(
        event_id=event_id,
        event_type=event_type,
        mode=mode,
        timestamp=timestamp,
        profile="browser-zh-melo",
        language="zh",
        tts_backend="local-http-wav",
        tts_label="local-http-wav/MeloTTS",
        vision_backend="moondream",
        source="test",
        session_id="session-one",
        client_id="client-one",
        metadata={"step_count": event_id},
        reason_codes=[f"test:event_{event_id}"],
    ).as_dict()


def test_replay_actor_trace_reconstructs_timeline_counts_and_labels(tmp_path):
    trace = tmp_path / "actor-trace.jsonl"
    payloads = [
        _event_payload(
            1,
            ActorEventTypeV2.CONNECTED,
            ActorEventModeV2.CONNECTED,
            timestamp="2026-04-27T00:00:00+00:00",
        ),
        _event_payload(
            2,
            ActorEventTypeV2.SPEECH_STARTED,
            ActorEventModeV2.LISTENING,
            timestamp="2026-04-27T00:00:01+00:00",
        ),
        _event_payload(
            3,
            ActorEventTypeV2.SPEAKING,
            ActorEventModeV2.SPEAKING,
            timestamp="2026-04-27T00:00:02+00:00",
        ),
    ]
    trace.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in payloads),
        encoding="utf-8",
    )

    summary = replay_actor_trace(trace)

    assert summary["event_count"] == 3
    assert summary["mode_timeline"] == [
        {
            "event_id": 1,
            "timestamp": "2026-04-27T00:00:00+00:00",
            "mode": "connected",
            "event_type": "connected",
        },
        {
            "event_id": 2,
            "timestamp": "2026-04-27T00:00:01+00:00",
            "mode": "listening",
            "event_type": "speech_started",
        },
        {
            "event_id": 3,
            "timestamp": "2026-04-27T00:00:02+00:00",
            "mode": "speaking",
            "event_type": "speaking",
        },
    ]
    assert summary["mode_counts"] == {"connected": 1, "listening": 1, "speaking": 1}
    assert summary["event_type_counts"] == {
        "connected": 1,
        "speaking": 1,
        "speech_started": 1,
    }
    assert summary["profiles"] == ["browser-zh-melo"]
    assert summary["languages"] == ["zh"]
    assert summary["tts_backends"] == ["local-http-wav"]
    assert summary["tts_labels"] == ["local-http-wav/MeloTTS"]
    assert summary["vision_backends"] == ["moondream"]
    assert summary["first_timestamp"] == "2026-04-27T00:00:00+00:00"
    assert summary["last_timestamp"] == "2026-04-27T00:00:02+00:00"
    assert summary["safety_violations"] == []


def test_replay_reports_malformed_and_unsafe_lines_without_raw_payload_leakage(tmp_path):
    trace = tmp_path / "actor-trace-unsafe.jsonl"
    safe_event = _event_payload(
        1,
        ActorEventTypeV2.CONNECTED,
        ActorEventModeV2.CONNECTED,
        timestamp="2026-04-27T00:00:00+00:00",
    )
    unsafe_event = dict(safe_event)
    unsafe_event["event_id"] = 2
    unsafe_event["metadata"] = {
        "safe_state": "ready",
        "raw_text": "private transcript",
        "safe_value_state": "a=candidate:private-candidate",
    }
    trace.write_text(
        "\n".join(
            [
                json.dumps(safe_event, ensure_ascii=False),
                "{not-json",
                json.dumps(unsafe_event, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = replay_actor_trace(trace)

    assert summary["event_count"] == 2
    assert summary["safety_violations"]
    encoded = json.dumps(summary, ensure_ascii=False)
    assert "actor_trace:malformed_json" in encoded
    assert "actor_trace:metadata_not_trace_safe" in encoded
    assert "private transcript" not in encoded
    assert "private-candidate" not in encoded


def test_replay_actor_trace_cli_prints_json_without_runtime_calls(tmp_path):
    trace = tmp_path / "actor-trace-cli.jsonl"
    trace.write_text(
        json.dumps(
            _event_payload(
                1,
                ActorEventTypeV2.WAITING,
                ActorEventModeV2.WAITING,
                timestamp="2026-04-27T00:00:00+00:00",
            ),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["BLINK_IMPORT_BANNER"] = "0"
    result = subprocess.run(
        [sys.executable, "scripts/evals/replay-actor-trace.py", str(trace)],
        cwd=os.getcwd(),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["event_count"] == 1
    assert payload["event_type_counts"] == {"waiting": 1}
    assert payload["mode_counts"] == {"waiting": 1}
    assert result.stderr == ""


def test_actor_trace_writer_writes_one_limit_event_then_stops(tmp_path):
    writer = ActorTraceWriter(trace_dir=tmp_path, profile="browser-en-kokoro", run_id="test", max_events=2)
    for event_id in range(1, 5):
        writer.append(
            ActorEventV2(
                event_id=event_id,
                event_type=ActorEventTypeV2.WAITING,
                mode=ActorEventModeV2.WAITING,
                profile="browser-en-kokoro",
                language="en",
                tts_backend="kokoro",
                tts_label="kokoro/English",
                vision_backend="moondream",
                source="test",
            )
        )

    lines = writer.path.read_text(encoding="utf-8").splitlines()
    payloads = [json.loads(line) for line in lines]
    assert len(payloads) == 3
    assert [payload["event_type"] for payload in payloads] == ["waiting", "waiting", "degraded"]
    assert payloads[-1]["reason_codes"] == ["trace.limit_reached"]


def test_browser_performance_event_bus_sinks_fail_open():
    class RaisingActorTraceWriter:
        def append(self, _event):
            raise RuntimeError("trace writer failed")

    class RaisingEpisodeWriter:
        def append(self, _event, *, terminal_event_type=None):
            raise RuntimeError("episode writer failed")

    class RaisingCollector:
        def append(self, _event, *, terminal_event_type=None):
            raise RuntimeError("collector failed")

    class RaisingScheduler:
        def observe_actor_event(self, _event):
            raise RuntimeError("scheduler failed")

    bus = BrowserPerformanceEventBus(
        actor_trace_writer=RaisingActorTraceWriter(),
        performance_episode_writer=RaisingEpisodeWriter(),
        discourse_episode_collector=RaisingCollector(),
        actor_control_scheduler=RaisingScheduler(),
    )

    event = bus.emit(
        event_type="browser_media.reported",
        source="test",
        mode=BrowserInteractionMode.CONNECTED,
        metadata={"camera_state": "permission_denied"},
        reason_codes=("browser_media:reported",),
    )

    assert event.event_type == "browser_media.reported"
    assert bus.latest_event_id == 1
    assert bus.actor_latest_event_id == 1
    assert bus.actor_control_latest_frame is None

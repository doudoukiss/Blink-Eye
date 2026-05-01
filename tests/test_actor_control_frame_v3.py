import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from blink.brain.persona import BrainExpressionVoicePolicy
from blink.brain.processors import BrainExpressionVoicePolicyProcessor
from blink.frames.frames import (
    AggregatedTextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from blink.interaction import (
    ACTOR_CONTROL_FRAME_V3_BOUNDARY_TYPES,
    ActorControlScheduler,
    BrowserInteractionMode,
    BrowserPerformanceEventBus,
    compile_actor_control_frames_v3,
    find_actor_control_safety_violations,
    load_actor_events_for_control_v3,
    render_actor_control_frames_v3_jsonl,
    replay_actor_control_frames_v3_jsonl,
    summarize_actor_control_frames_v3,
)
from blink.processors.frame_processor import FrameDirection

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "actor_control_frame_v3.schema.json"


def _schema_validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _assert_schema_valid(payload: dict[str, object]) -> None:
    errors = sorted(_schema_validator().iter_errors(payload), key=lambda error: error.path)
    assert errors == []


def _event(
    event_id: int,
    event_type: str,
    *,
    mode: str,
    profile: str = "browser-zh-melo",
    language: str = "zh",
    tts_backend: str = "local-http-wav",
    tts_label: str = "local-http-wav/MeloTTS",
    source: str = "test",
    metadata: dict[str, object] | None = None,
    reason_codes: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "event_id": event_id,
        "event_type": event_type,
        "mode": mode,
        "timestamp": f"2026-04-27T00:00:{event_id:02d}+00:00",
        "profile": profile,
        "language": language,
        "tts_backend": tts_backend,
        "tts_label": tts_label,
        "vision_backend": "moondream",
        "source": source,
        "session_id": "session-1",
        "client_id": "client-1",
        "metadata": metadata or {},
        "reason_codes": list(reason_codes),
    }


def _matched_events(*, profile: str, language: str, tts_backend: str, tts_label: str):
    labels = {
        "profile": profile,
        "language": language,
        "tts_backend": tts_backend,
        "tts_label": tts_label,
    }
    return [
        _event(1, "speech_started", mode="listening", reason_codes=("voice:speech_started",), **labels),
        _event(
            2,
            "final_heard",
            mode="heard",
            metadata={
                "semantic_listener": {
                    "schema_version": 3,
                    "detected_intent": "project_planning",
                    "listener_chip_ids": [
                        "heard_summary",
                        "constraint_detected",
                        "ready_to_answer",
                    ],
                    "listener_chip_count": 3,
                    "camera_reference_state": "stale_or_limited",
                    "memory_context_state": "available",
                    "floor_state": "handoff_pending",
                    "enough_information_to_answer": True,
                    "summary_hash": "0123456789abcdef",
                    "reason_codes": ["semantic_listener:v3"],
                }
            },
            reason_codes=("stt:transcribed", "active_listener:final_understanding_ready"),
            **labels,
        ),
        _event(
            3,
            "speaking",
            mode="speaking",
            metadata={
                "generation_id": "speech-1",
                "turn_id": "turn-1",
                "chunk_index": 1,
                "queue_depth": 1,
            },
            reason_codes=("speech:subtitle_ready",),
            **labels,
        ),
        _event(4, "waiting", mode="waiting", reason_codes=("tts:stopped",), **labels),
        _event(
            5,
            "looking",
            mode="looking",
            source="vision",
            metadata={
                "frame_seq": 9,
                "frame_age_ms": 40,
                "scene_transition": "vision_answered",
                "camera_honesty_state": "can_see_now",
                "user_presence_hint": "present",
                "last_moondream_result_state": "answered",
                "object_showing_likelihood": 0.85,
            },
            reason_codes=("vision:fetch_user_image_success",),
            **labels,
        ),
        _event(6, "memory_used", mode="thinking", reason_codes=("memory:used",), **labels),
        _event(
            7,
            "persona_plan_compiled",
            mode="thinking",
            metadata={"persona_reference_count": 2},
            reason_codes=("persona:plan_compiled",),
            **labels,
        ),
        _event(
            8,
            "interruption_accepted",
            mode="interrupted",
            reason_codes=("interruption:accepted",),
            **labels,
        ),
        _event(
            9,
            "output_flushed",
            mode="interrupted",
            metadata={"output_flushed_count": 1},
            reason_codes=("interruption:output_flushed",),
            **labels,
        ),
    ]


def test_actor_control_frame_schema_valid_and_boundary_mapping():
    frames = compile_actor_control_frames_v3(
        _matched_events(
            profile="browser-zh-melo",
            language="zh",
            tts_backend="local-http-wav",
            tts_label="local-http-wav/MeloTTS",
        )
    )
    payloads = [frame.as_dict() for frame in frames]
    boundaries = [payload["boundary"] for payload in payloads]

    for payload in payloads:
        _assert_schema_valid(payload)
        assert payload["schema_version"] == 3
        assert "manual_boundary" not in json.dumps(payload)

    assert set(ACTOR_CONTROL_FRAME_V3_BOUNDARY_TYPES) <= set(boundaries)
    assert boundaries[:4] == [
        "vad_boundary",
        "stt_final_boundary",
        "speech_chunk_boundary",
        "tts_queue_boundary",
    ]
    assert payloads[4]["camera_policy"]["fresh_frame_used"] is True
    assert payloads[4]["camera_policy"]["scene_transition"] == "vision_answered"
    assert payloads[4]["camera_policy"]["camera_honesty_state"] == "can_see_now"
    assert payloads[4]["camera_policy"]["last_moondream_result_state"] == "answered"
    assert payloads[4]["camera_policy"]["object_showing_likelihood"] == 0.85
    assert payloads[1]["persistent_state"]["active_listener_intent"] == "project_planning"
    assert payloads[1]["persistent_state"]["active_listener_chip_ids"] == [
        "heard_summary",
        "constraint_detected",
        "ready_to_answer",
    ]
    assert payloads[1]["active_listener_policy"]["detected_intent"] == "project_planning"
    assert payloads[1]["active_listener_policy"]["listener_chip_ids"] == [
        "heard_summary",
        "constraint_detected",
        "ready_to_answer",
    ]
    assert payloads[-1]["repair_policy"]["stale_output_action"] == "dropped_or_suppressed"


def test_primary_profiles_have_structural_parity_except_labels():
    zh = [
        frame.as_dict()
        for frame in compile_actor_control_frames_v3(
            _matched_events(
                profile="browser-zh-melo",
                language="zh",
                tts_backend="local-http-wav",
                tts_label="local-http-wav/MeloTTS",
            )
        )
    ]
    en = [
        frame.as_dict()
        for frame in compile_actor_control_frames_v3(
            _matched_events(
                profile="browser-en-kokoro",
                language="en",
                tts_backend="kokoro",
                tts_label="kokoro/English",
            )
        )
    ]

    assert [frame["boundary"] for frame in zh] == [frame["boundary"] for frame in en]
    assert [set(frame) for frame in zh] == [set(frame) for frame in en]
    assert [set(frame["speech_policy"]) for frame in zh] == [
        set(frame["speech_policy"]) for frame in en
    ]
    assert zh[0]["profile"] == "browser-zh-melo"
    assert en[0]["profile"] == "browser-en-kokoro"
    assert zh[0]["language"] == "zh"
    assert en[0]["language"] == "en"
    assert zh[0]["tts_runtime_label"] == "local-http-wav/MeloTTS"
    assert en[0]["tts_runtime_label"] == "kokoro/English"


def test_floor_sub_state_is_carried_in_control_frame_and_replay_summary():
    frames = compile_actor_control_frames_v3(
        [
            _event(
                1,
                "floor_transition",
                mode="interrupted",
                source="floor",
                metadata={
                    "floor_state": "repair",
                    "floor_sub_state": "repair_requested",
                    "yield_decision": "yield_to_user",
                },
                reason_codes=("floor:explicit_interruption", "explicit_interrupt"),
            )
        ]
    )
    payload = frames[0].as_dict()
    summary = summarize_actor_control_frames_v3([payload]).as_dict()

    _assert_schema_valid(payload)
    assert payload["persistent_state"]["floor_sub_state"] == "repair_requested"
    assert payload["floor_policy"]["sub_state"] == "repair_requested"
    assert payload["floor_policy"]["yield_decision"] == "yield_to_user"
    assert summary["floor_sub_states"] == ["repair_requested"]
    assert summary["boundary_timeline"][0]["floor_sub_state"] == "repair_requested"


def test_camera_stale_and_error_do_not_report_fresh_use():
    stale, error = compile_actor_control_frames_v3(
        [
            _event(
                1,
                "degraded",
                mode="degraded",
                source="camera",
                metadata={"camera_state": "stale", "scene_transition": "vision_stale"},
                reason_codes=("camera:frame_stale",),
            ),
            _event(
                2,
                "error",
                mode="error",
                source="vision",
                metadata={
                    "camera_state": "error",
                    "scene_transition": "vision_unavailable",
                },
                reason_codes=("vision:last_result_error",),
            ),
        ]
    )

    assert stale.camera_policy["state"] == "stale_or_limited"
    assert stale.camera_policy["fresh_frame_used"] is False
    assert stale.camera_policy["scene_transition"] == "vision_stale"
    assert stale.camera_policy["camera_honesty_state"] == "unavailable"
    assert error.camera_policy["state"] == "error"
    assert error.camera_policy["fresh_frame_used"] is False
    assert error.camera_policy["scene_transition"] == "vision_unavailable"


def test_replay_is_deterministic_from_control_jsonl(tmp_path):
    frames = compile_actor_control_frames_v3(
        _matched_events(
            profile="browser-en-kokoro",
            language="en",
            tts_backend="kokoro",
            tts_label="kokoro/English",
        )
    )
    path = tmp_path / "control.jsonl"
    path.write_text(render_actor_control_frames_v3_jsonl(frames), encoding="utf-8")

    first = replay_actor_control_frames_v3_jsonl(path).as_dict()
    second = replay_actor_control_frames_v3_jsonl(path).as_dict()

    assert first == second
    assert first["frame_count"] == len(frames)
    assert first["profiles"] == ["browser-en-kokoro"]
    assert first["boundary_counts"]["repair_boundary"] == 1
    assert "speech:generation_stale" in first["failure_labels"]


def test_unsafe_input_fails_closed_without_echoing_values(tmp_path):
    path = tmp_path / "unsafe.jsonl"
    path.write_text(
        json.dumps(
            {
                **_event(1, "speech_started", mode="listening"),
                "metadata": {
                    "raw_audio": "data:audio/wav;base64,SECRET",
                    "safe_count": 1,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    events, violations = load_actor_events_for_control_v3(path)
    encoded = json.dumps({"events": events, "violations": violations}, ensure_ascii=False)
    direct = find_actor_control_safety_violations(json.loads(path.read_text(encoding="utf-8")))

    assert violations
    assert direct
    assert "SECRET" not in encoded
    assert "data:audio" not in encoded

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evals/replay-actor-control-frame-v3.py",
            str(path),
            "--input-format",
            "actor-trace",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "SECRET" not in result.stdout
    assert "actor_control:unsafe_metadata_key_present" in result.stdout


def _policy() -> BrainExpressionVoicePolicy:
    return BrainExpressionVoicePolicy(
        available=True,
        modality="browser",
        concise_chunking_active=False,
        chunking_mode="off",
        max_spoken_chunk_chars=0,
        interruption_strategy_label="protected",
        pause_yield_hint="metadata only",
        active_hints=(),
        unsupported_hints=("pause_timing",),
        noop_reason_codes=("voice_policy_noop:pause_timing_metadata_only",),
        expression_controls_hardware=False,
        reason_codes=("voice_policy:available",),
    )


class CaptureSpeechPolicyProcessor(BrainExpressionVoicePolicyProcessor):
    def __init__(self, *, bus: BrowserPerformanceEventBus, scheduler: ActorControlScheduler):
        super().__init__(
            policy_provider=_policy,
            tts_backend="kokoro",
            language="en",
            speech_director_mode="kokoro_chunked",
            performance_emit=bus.emit,
            actor_control_scheduler=scheduler,
            enable_direct_mode=True,
        )
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


@pytest.mark.asyncio
async def test_speech_lookahead_gates_and_drains_on_tts_queue_boundary():
    scheduler = ActorControlScheduler(
        profile="browser-en-kokoro",
        language="en",
        tts_backend="kokoro",
        tts_runtime_label="kokoro/English",
    )
    bus = BrowserPerformanceEventBus(
        max_events=50,
        actor_context_provider=lambda: {
            "profile": "browser-en-kokoro",
            "language": "en",
            "tts_backend": "kokoro",
            "tts_label": "kokoro/English",
            "vision_backend": "moondream",
        },
        actor_control_scheduler=scheduler,
    )
    processor = CaptureSpeechPolicyProcessor(bus=bus, scheduler=scheduler)
    text = (
        "First sentence is ready and has enough detail for a separate chunk. "
        "Second sentence is ready and has enough detail for another chunk. "
        "Third sentence waits because the lookahead cap is already full. "
        "Fourth sentence waits until a TTS queue boundary releases capacity. "
        "Fifth sentence confirms the held buffer remains interruptible."
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    events = bus.actor_payload(limit=50)["events"]

    assert len(chunks) == 2
    assert any("speech:lookahead_held" in event["reason_codes"] for event in events)
    assert bus.actor_control_latest_frame is not None
    assert bus.actor_control_latest_frame.lookahead_counters["speech_chunks_outstanding"] == 2
    assert bus.actor_control_latest_frame.speech_policy["estimated_duration_ms"] > 0
    assert (
        bus.actor_control_latest_frame.speech_policy["subtitle_timing_policy"]
        == "before_or_at_playback_start"
    )
    assert (
        "speech_capability:supports_chunk_boundaries:supported"
        in bus.actor_control_latest_frame.speech_policy["backend_capability_labels"]
    )
    assert (
        "speech_capability:supports_speech_rate:noop"
        in bus.actor_control_latest_frame.speech_policy["backend_capability_labels"]
    )

    bus.emit(
        event_type="tts.speech_end",
        source="tts",
        mode=BrowserInteractionMode.WAITING,
        reason_codes=("tts:stopped",),
    )
    await processor.drain_held_speech_chunks()
    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]

    assert len(chunks) == 3
    assert bus.actor_control_latest_frame.boundary == "speech_chunk_boundary"

    bus.emit(
        event_type="tts.speech_end",
        source="tts",
        mode=BrowserInteractionMode.WAITING,
        reason_codes=("tts:stopped",),
    )
    await processor.drain_held_speech_chunks()
    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]

    assert len(chunks) == 4
    assert bus.actor_control_latest_frame.boundary == "speech_chunk_boundary"


@pytest.mark.asyncio
async def test_speech_lookahead_drain_releases_short_final_tail_after_llm_end():
    scheduler = ActorControlScheduler(
        profile="browser-en-kokoro",
        language="en",
        tts_backend="kokoro",
        tts_runtime_label="kokoro/English",
    )
    bus = BrowserPerformanceEventBus(
        max_events=50,
        actor_context_provider=lambda: {
            "profile": "browser-en-kokoro",
            "language": "en",
            "tts_backend": "kokoro",
            "tts_label": "kokoro/English",
            "vision_backend": "moondream",
        },
        actor_control_scheduler=scheduler,
    )
    processor = CaptureSpeechPolicyProcessor(bus=bus, scheduler=scheduler)
    text = (
        "First sentence is ready and has enough detail for a separate chunk "
        "that should start this turn now. "
        "Second sentence is ready and has enough detail for another chunk "
        "that should stay bounded now. Ok"
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    end_frames = [
        frame for frame, _ in processor.pushed if isinstance(frame, LLMFullResponseEndFrame)
    ]

    assert [frame.text for frame in chunks] == [
        "First sentence is ready and has enough detail for a separate chunk "
        "that should start this turn now.",
        "Second sentence is ready and has enough detail for another chunk "
        "that should stay bounded now.",
    ]
    assert not end_frames

    bus.emit(
        event_type="tts.speech_end",
        source="tts",
        mode=BrowserInteractionMode.WAITING,
        reason_codes=("tts:stopped",),
    )
    await processor.drain_held_speech_chunks()

    chunks = [frame for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    end_frames = [
        frame for frame, _ in processor.pushed if isinstance(frame, LLMFullResponseEndFrame)
    ]
    assert [frame.text for frame in chunks][-1] == "Ok"
    assert end_frames


def test_summarize_actor_control_frames_reports_memory_persona_effects():
    frames = compile_actor_control_frames_v3(
        _matched_events(
            profile="browser-zh-melo",
            language="zh",
            tts_backend="local-http-wav",
            tts_label="local-http-wav/MeloTTS",
        )
    )
    summary = summarize_actor_control_frames_v3(frame.as_dict() for frame in frames).as_dict()

    assert summary["memory_persona_effects"]["memory_effect_count"] == 1
    assert summary["memory_persona_effects"]["persona_effect_count"] == 2
    assert summary["performance_plan_summaries"]
    assert {
        "sequence",
        "boundary",
        "stance",
        "response_shape",
        "plan_summary",
    } <= set(summary["performance_plan_summaries"][0])
    assert "accepted" in summary["interruption_outcomes"]
    assert "handoff_complete" in summary["floor_sub_states"]

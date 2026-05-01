import json

from blink.brain.identity import base_brain_system_prompt
from blink.brain.persona import (
    BrainExpressionVoiceMetricsRecorder,
    BrainExpressionVoicePolicy,
    unavailable_expression_voice_metrics_snapshot,
)
from blink.brain.processors import _next_voice_chunk
from blink.brain.runtime import BrainRuntime
from blink.brain.session import resolve_brain_session_ids
from blink.transcriptions.language import Language


class DummyLLM:
    def register_function(self, function_name, handler):
        return None


def _policy(*, concise: bool = True) -> BrainExpressionVoicePolicy:
    return BrainExpressionVoicePolicy(
        available=True,
        modality="voice",
        concise_chunking_active=concise,
        chunking_mode="concise" if concise else "off",
        max_spoken_chunk_chars=42 if concise else 220,
        interruption_strategy_label="yield after brief pause",
        pause_yield_hint="pause=0.32; yield=yield after brief pause",
        active_hints=("concise_chunking",) if concise else (),
        unsupported_hints=("speech_rate", "prosody_emphasis", "pause_timing"),
        noop_reason_codes=(
            "voice_policy_noop:speech_rate:provider-neutral",
            "voice_policy_noop:prosody:provider-neutral",
        ),
        expression_controls_hardware=False,
        reason_codes=("voice_policy:available",),
    )


def test_voice_metrics_snapshot_serialization_is_stable_and_bounded():
    recorder = BrainExpressionVoiceMetricsRecorder()

    recorder.record_response_start(_policy())
    recorder.record_chunk("short chunk")
    recorder.record_chunk("a longer chunk")
    recorder.record_buffer_flush(emitted_chunk_count=2)
    recorder.record_interruption(discarded_buffer=True)
    recorder.record_first_subtitle_latency(40.0)
    recorder.record_first_audio_latency(120.0)
    recorder.record_speech_chunk_latency(30.0)
    recorder.record_speech_queue_depth(2)
    recorder.record_stale_chunk_drop()
    recorder.record_resumed_latency_after_interrupt(80.0)
    recorder.record_partial_stream_abort()
    snapshot = recorder.snapshot()

    assert snapshot.available is True
    assert snapshot.response_count == 1
    assert snapshot.concise_chunking_activation_count == 1
    assert snapshot.chunk_count == 2
    assert snapshot.max_chunk_chars == len("a longer chunk")
    assert snapshot.average_chunk_chars == 12.5
    assert snapshot.interruption_frame_count == 1
    assert snapshot.interruption_accept_count == 1
    assert snapshot.buffer_discard_count == 1
    assert snapshot.first_audio_latency_ms == 120.0
    assert snapshot.first_audio_latency_sample_count == 1
    assert snapshot.first_subtitle_latency_ms == 40.0
    assert snapshot.first_subtitle_latency_sample_count == 1
    assert snapshot.speech_chunk_latency_ms == 30.0
    assert snapshot.speech_chunk_latency_sample_count == 1
    assert snapshot.speech_queue_depth_current == 2
    assert snapshot.speech_queue_depth_max == 2
    assert snapshot.stale_chunk_drop_count == 1
    assert snapshot.resumed_latency_after_interrupt_ms == 80.0
    assert snapshot.resumed_latency_sample_count == 1
    assert snapshot.partial_stream_abort_count == 1
    assert snapshot.average_chunks_per_response == 2.0
    assert snapshot.p50_chunk_chars == len("short chunk")
    assert snapshot.p95_chunk_chars == len("a longer chunk")
    assert snapshot.expression_controls_hardware is False
    assert json.loads(json.dumps(snapshot.as_dict(), sort_keys=True)) == snapshot.as_dict()


def test_voice_metrics_recorder_reset_is_deterministic():
    recorder = BrainExpressionVoiceMetricsRecorder()

    recorder.record_response_start(_policy())
    recorder.record_chunk("chunk")
    recorder.record_buffer_flush(emitted_chunk_count=1)
    recorder.reset()
    snapshot = recorder.snapshot()

    assert snapshot.available is True
    assert snapshot.response_count == 0
    assert snapshot.concise_chunking_activation_count == 0
    assert snapshot.chunk_count == 0
    assert snapshot.max_chunk_chars == 0
    assert snapshot.average_chunk_chars == 0.0
    assert snapshot.first_audio_latency_ms == 0.0
    assert snapshot.first_subtitle_latency_ms == 0.0
    assert snapshot.speech_chunk_latency_ms == 0.0
    assert snapshot.speech_queue_depth_current == 0
    assert snapshot.stale_chunk_drop_count == 0
    assert snapshot.resumed_latency_after_interrupt_ms == 0.0
    assert snapshot.partial_stream_abort_count == 0
    assert snapshot.average_chunks_per_response == 0.0
    assert snapshot.p50_chunk_chars == 0
    assert snapshot.p95_chunk_chars == 0
    assert snapshot.last_chunking_mode == "none"
    assert snapshot.last_max_spoken_chunk_chars == 0


def test_voice_metrics_unavailable_snapshot_is_safe():
    snapshot = unavailable_expression_voice_metrics_snapshot("runtime_not_active")
    payload = snapshot.as_dict()

    assert payload["available"] is False
    assert payload["response_count"] == 0
    assert payload["expression_controls_hardware"] is False
    assert "runtime_not_active" in payload["reason_codes"]
    assert "voice_policy_noop:hardware_control_forbidden" in payload["reason_codes"]
    assert "speech_rate_supported" not in json.dumps(payload, sort_keys=True)
    assert "prosody_supported" not in json.dumps(payload, sort_keys=True)


def test_voice_chunking_waits_for_chinese_sentence_boundary_not_comma():
    text = "我镜头前是一位戴眼镜的人，正坐在书架前。后面还有内容"

    chunk, remaining = _next_voice_chunk(text, limit=80, force=False)

    assert chunk == "我镜头前是一位戴眼镜的人，正坐在书架前。"
    assert remaining == "后面还有内容"


def test_voice_chunking_does_not_flush_short_clause_at_english_comma():
    text = "I can see the bookshelf, and you are seated in front of it"

    chunk, remaining = _next_voice_chunk(text, limit=80, force=False)

    assert chunk is None
    assert remaining == text


def test_voice_chunking_can_hold_short_sentences_for_slow_local_tts():
    text = "好的。我们一步一步来。先看麦克风，再看摄像头。然后判断瓶颈。"

    chunk, remaining = _next_voice_chunk(text, limit=80, force=False, min_chars=24)

    assert chunk == "好的。我们一步一步来。先看麦克风，再看摄像头。然后判断瓶颈。"
    assert remaining == ""


def test_voice_chunking_waits_when_first_sentence_is_below_minimum():
    text = "好的。"

    chunk, remaining = _next_voice_chunk(text, limit=80, force=False, min_chars=24)

    assert chunk is None
    assert remaining == text


def test_brain_runtime_exposes_default_voice_metrics(tmp_path):
    runtime = BrainRuntime(
        base_prompt=base_brain_system_prompt(Language.ZH),
        language=Language.ZH,
        runtime_kind="voice",
        session_resolver=lambda: resolve_brain_session_ids(
            runtime_kind="voice",
            client_id="voice-metrics-runtime",
        ),
        llm=DummyLLM(),
        brain_db_path=tmp_path / "brain.db",
        tts_backend="local-http-wav",
    )
    try:
        initial = runtime.current_voice_metrics()
        runtime.voice_metrics_recorder.record_response_start(_policy(concise=False))
        updated = runtime.current_voice_metrics()
    finally:
        runtime.close()

    assert initial.available is True
    assert initial.response_count == 0
    assert initial.expression_controls_hardware is False
    assert updated.response_count == 1
    assert updated.concise_chunking_activation_count == 0
    assert updated.last_chunking_mode == "off"
    assert updated.expression_controls_hardware is False

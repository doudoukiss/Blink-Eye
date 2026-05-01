import pytest

from blink.brain.identity import load_default_agent_blocks
from blink.brain.persona import (
    BrainExpressionVoiceMetricsRecorder,
    BrainExpressionVoicePolicy,
    BrainPersonaModality,
    BrainPersonaTaskMode,
    BrainVoiceBackendCapabilities,
    BrainVoiceBackendCapabilityRegistry,
    compile_expression_frame,
    compile_expression_voice_policy,
    compile_persona_frame,
    compile_realtime_voice_actuation_plan,
    provider_neutral_voice_capabilities,
    resolve_voice_backend_capabilities,
)
from blink.brain.processors import BrainExpressionVoicePolicyProcessor
from blink.frames.frames import (
    AggregatedTextFrame,
    AggregationType,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from blink.processors.frame_processor import FrameDirection
from blink.transcriptions.language import Language


def _expression(*, modality=BrainPersonaModality.VOICE, seriousness="normal"):
    persona_frame = compile_persona_frame(
        agent_blocks=load_default_agent_blocks(),
        task_mode=BrainPersonaTaskMode.REPLY,
        modality=modality,
    )
    return compile_expression_frame(
        persona_frame=persona_frame,
        relationship_style=None,
        teaching_profile=None,
        task_mode=BrainPersonaTaskMode.REPLY,
        modality=modality,
        language=Language.ZH,
        seriousness=seriousness,
    )


def _inactive_policy() -> BrainExpressionVoicePolicy:
    return compile_expression_voice_policy(None, modality=BrainPersonaModality.VOICE)


def _concise_policy(*, max_chars: int = 42) -> BrainExpressionVoicePolicy:
    policy = compile_expression_voice_policy(
        _expression(modality=BrainPersonaModality.VOICE, seriousness="safety"),
        modality=BrainPersonaModality.VOICE,
        tts_backend="local-http-wav",
    )
    return BrainExpressionVoicePolicy(
        available=policy.available,
        modality=policy.modality,
        concise_chunking_active=True,
        chunking_mode=policy.chunking_mode,
        max_spoken_chunk_chars=max_chars,
        interruption_strategy_label=policy.interruption_strategy_label,
        pause_yield_hint=policy.pause_yield_hint,
        active_hints=policy.active_hints,
        unsupported_hints=policy.unsupported_hints,
        noop_reason_codes=policy.noop_reason_codes,
        expression_controls_hardware=policy.expression_controls_hardware,
        reason_codes=policy.reason_codes,
    )


class CaptureVoicePolicyProcessor(BrainExpressionVoicePolicyProcessor):
    def __init__(
        self,
        policy: BrainExpressionVoicePolicy,
        *,
        capabilities: BrainVoiceBackendCapabilities | None = None,
        tts_backend: str | None = None,
        metrics_recorder: BrainExpressionVoiceMetricsRecorder | None = None,
    ):
        super().__init__(
            policy_provider=lambda: policy,
            capabilities_provider=(lambda: capabilities) if capabilities is not None else None,
            tts_backend=tts_backend,
            metrics_recorder=metrics_recorder,
            enable_direct_mode=True,
        )
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


@pytest.mark.parametrize(
    "modality",
    [
        BrainPersonaModality.VOICE,
        BrainPersonaModality.BROWSER,
        BrainPersonaModality.EMBODIED,
    ],
)
def test_voice_like_expression_frames_produce_available_policy(modality):
    policy = compile_expression_voice_policy(
        _expression(modality=modality),
        modality=modality,
        tts_backend="local-http-wav",
    )

    assert policy.available is True
    assert policy.modality == modality.value
    assert policy.expression_controls_hardware is False
    assert "speech_rate" in policy.unsupported_hints
    assert "prosody_emphasis" in policy.unsupported_hints
    assert "voice_policy_noop:pause_timing_metadata_only" in policy.noop_reason_codes
    assert "voice_policy_noop:hardware_control_forbidden" in policy.noop_reason_codes


def test_text_or_missing_expression_policy_fails_closed():
    text_policy = compile_expression_voice_policy(
        _expression(modality=BrainPersonaModality.TEXT),
        modality=BrainPersonaModality.TEXT,
    )
    missing_policy = compile_expression_voice_policy(None, modality=BrainPersonaModality.VOICE)

    assert text_policy.available is False
    assert missing_policy.available is False
    assert text_policy.expression_controls_hardware is False
    assert missing_policy.expression_controls_hardware is False
    assert "voice_policy:unavailable" in text_policy.reason_codes
    assert "voice_policy_frame_missing" in missing_policy.reason_codes


def test_voice_backend_capability_matrix_is_conservative():
    capabilities = provider_neutral_voice_capabilities(backend_label="local-http-wav")
    payload = capabilities.as_dict()

    assert payload["backend_label"] == "local-http-wav"
    assert payload["supports_chunk_boundaries"] is True
    assert payload["supports_interruption_flush"] is True
    assert payload["supports_interruption_discard"] is False
    assert payload["supports_pause_timing"] is False
    assert payload["supports_speech_rate"] is False
    assert payload["supports_prosody_emphasis"] is False
    assert payload["supports_partial_stream_abort"] is False
    assert payload["expression_controls_hardware"] is False
    assert "voice_capability_noop:interruption_discard" in payload["reason_codes"]
    assert "voice_capability_noop:speech_rate" in payload["reason_codes"]
    assert "voice_capability_noop:hardware_control_forbidden" in payload["reason_codes"]


def test_voice_backend_registry_resolves_known_backends_and_fallback():
    known = resolve_voice_backend_capabilities("local-http-wav")
    kokoro = resolve_voice_backend_capabilities("kokoro")
    fallback = resolve_voice_backend_capabilities("unlisted-backend")

    assert known.fallback_used is False
    assert known.resolved_backend_label == "local-http-wav"
    assert known.capabilities.supports_chunk_boundaries is True
    assert known.capabilities.supports_interruption_flush is True
    assert known.capabilities.supports_interruption_discard is False
    assert known.capabilities.supports_speech_rate is False
    assert known.capabilities.supports_prosody_emphasis is False
    assert known.capabilities.supports_pause_timing is False
    assert known.capabilities.supports_partial_stream_abort is False
    assert known.capabilities.expression_controls_hardware is False
    assert "voice_backend_registry:known_backend" in known.reason_codes
    assert "voice_backend_registry:known_backend" in known.capabilities.reason_codes
    assert "voice_backend_registry:local_http_wav_melo" in known.capabilities.reason_codes

    assert kokoro.fallback_used is False
    assert kokoro.resolved_backend_label == "kokoro"
    assert kokoro.capabilities.supports_chunk_boundaries is True
    assert kokoro.capabilities.supports_interruption_flush is True
    assert kokoro.capabilities.supports_interruption_discard is False
    assert kokoro.capabilities.supports_speech_rate is False
    assert kokoro.capabilities.supports_prosody_emphasis is False
    assert kokoro.capabilities.supports_pause_timing is False
    assert kokoro.capabilities.supports_partial_stream_abort is False
    assert kokoro.capabilities.expression_controls_hardware is False
    assert "voice_backend_registry:known_backend" in kokoro.reason_codes
    assert "voice_backend_registry:kokoro_chunked" in kokoro.capabilities.reason_codes

    assert fallback.fallback_used is True
    assert fallback.resolved_backend_label == "unlisted-backend"
    assert fallback.capabilities.supports_chunk_boundaries is True
    assert fallback.capabilities.supports_partial_stream_abort is False
    assert "voice_backend_registry:fallback_provider_neutral" in fallback.reason_codes


def test_voice_backend_registry_accepts_explicit_capability_entries():
    registry = BrainVoiceBackendCapabilityRegistry.from_mapping(
        {
            "test-stream": BrainVoiceBackendCapabilities(
                backend_label="test-stream",
                supports_chunk_boundaries=True,
                supports_interruption_flush=True,
                supports_interruption_discard=True,
                supports_speech_rate=True,
                supports_prosody_emphasis=True,
                reason_codes=("test_stream_capabilities",),
            )
        }
    )

    resolution = resolve_voice_backend_capabilities("test-stream", registry=registry)

    assert resolution.fallback_used is False
    assert resolution.capabilities.supports_interruption_discard is True
    assert resolution.capabilities.supports_speech_rate is True
    assert resolution.capabilities.supports_pause_timing is False
    assert resolution.capabilities.expression_controls_hardware is False


def test_voice_actuation_plan_serializes_supported_and_noop_hints():
    policy = compile_expression_voice_policy(
        _expression(modality=BrainPersonaModality.VOICE, seriousness="safety"),
        modality=BrainPersonaModality.VOICE,
        tts_backend="local-http-wav",
    )
    plan = compile_realtime_voice_actuation_plan(
        policy,
        capabilities=provider_neutral_voice_capabilities(backend_label="local-http-wav"),
        tts_backend="local-http-wav",
    )
    payload = plan.as_dict()

    assert payload["available"] is True
    assert payload["backend_label"] == "local-http-wav"
    assert payload["chunk_boundaries_enabled"] is True
    assert payload["interruption_flush_enabled"] is True
    assert payload["interruption_discard_enabled"] is False
    assert payload["speech_rate_enabled"] is False
    assert payload["prosody_emphasis_enabled"] is False
    assert payload["pause_timing_enabled"] is False
    assert payload["partial_stream_abort_enabled"] is False
    assert payload["expression_controls_hardware"] is False
    assert "speech_rate" in payload["requested_hints"]
    assert "prosody_emphasis" in payload["requested_hints"]
    assert "pause_timing" in payload["requested_hints"]
    assert "partial_stream_abort" in payload["requested_hints"]
    assert "chunk_boundaries" in payload["applied_hints"]
    assert "interruption_flush" in payload["applied_hints"]
    assert "speech_rate" not in payload["applied_hints"]
    assert "speech_rate" in payload["unsupported_hints"]
    assert "partial_stream_abort" in payload["unsupported_hints"]
    assert "voice_actuation_noop:interruption_discard_unsupported" in payload["noop_reason_codes"]
    assert "voice_actuation_noop:speech_rate_unsupported" in payload["noop_reason_codes"]
    assert "voice_actuation_noop:prosody_emphasis_unsupported" in payload["noop_reason_codes"]


def test_voice_actuation_plan_records_backend_supported_applied_hints():
    policy = compile_expression_voice_policy(
        _expression(modality=BrainPersonaModality.VOICE, seriousness="safety"),
        modality=BrainPersonaModality.VOICE,
        tts_backend="test-stream",
    )
    plan = compile_realtime_voice_actuation_plan(
        policy,
        capabilities=BrainVoiceBackendCapabilities(
            backend_label="test-stream",
            supports_chunk_boundaries=True,
            supports_interruption_flush=True,
            supports_interruption_discard=True,
            supports_pause_timing=False,
            supports_speech_rate=True,
            supports_prosody_emphasis=True,
            supports_partial_stream_abort=False,
            reason_codes=("test_stream_capabilities",),
        ),
        tts_backend="test-stream",
    )
    payload = plan.as_dict()

    assert payload["backend_label"] == "test-stream"
    assert payload["speech_rate_enabled"] is True
    assert payload["prosody_emphasis_enabled"] is True
    assert payload["pause_timing_enabled"] is False
    assert payload["partial_stream_abort_enabled"] is False
    assert "speech_rate" in payload["requested_hints"]
    assert "speech_rate" in payload["applied_hints"]
    assert "prosody_emphasis" in payload["applied_hints"]
    assert "interruption_discard" in payload["applied_hints"]
    assert "pause_timing" in payload["unsupported_hints"]
    assert "partial_stream_abort" in payload["unsupported_hints"]
    assert "speech_rate" not in payload["unsupported_hints"]
    assert payload["expression_controls_hardware"] is False


def test_serious_expression_uses_shorter_concise_chunk_policy():
    normal_expression = _expression(modality=BrainPersonaModality.VOICE, seriousness="normal")
    safety_expression = _expression(modality=BrainPersonaModality.VOICE, seriousness="safety")
    normal_policy = compile_expression_voice_policy(
        normal_expression,
        modality=BrainPersonaModality.VOICE,
    )
    safety_policy = compile_expression_voice_policy(
        safety_expression,
        modality=BrainPersonaModality.VOICE,
    )

    assert safety_expression.humor_budget < normal_expression.humor_budget
    assert safety_expression.voice_hints is not None
    assert normal_expression.voice_hints is not None
    assert (
        safety_expression.voice_hints.excitement_ceiling
        < normal_expression.voice_hints.excitement_ceiling
    )
    assert safety_policy.concise_chunking_active is True
    assert safety_policy.max_spoken_chunk_chars < normal_policy.max_spoken_chunk_chars
    assert safety_policy.chunking_mode == "safety_concise"


@pytest.mark.asyncio
async def test_inactive_voice_policy_processor_passes_text_through():
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(_inactive_policy(), metrics_recorder=recorder)
    text_frame = LLMTextFrame("A normal spoken sentence.")

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(text_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    metrics = recorder.snapshot()

    assert text_frame in [frame for frame, _direction in processor.pushed]
    assert not any(isinstance(frame, AggregatedTextFrame) for frame, _direction in processor.pushed)
    assert metrics.response_count == 1
    assert metrics.concise_chunking_activation_count == 0
    assert metrics.chunk_count == 0
    assert metrics.last_chunking_mode == "unavailable"


@pytest.mark.asyncio
async def test_concise_voice_policy_processor_splits_english_chunks():
    policy = _concise_policy(max_chars=52)
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(policy, metrics_recorder=recorder)
    text = (
        "First, isolate the failing step and keep the repro small. "
        "Then change one thing at a time and observe the result. "
        "Finally, summarize the evidence before trying a fix."
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    chunks = [frame.text for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    metrics = recorder.snapshot()
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
    assert all(len(chunk) <= policy.max_spoken_chunk_chars for chunk in chunks)
    assert all(
        frame.aggregated_by is AggregationType.SENTENCE
        for frame, _ in processor.pushed
        if isinstance(frame, AggregatedTextFrame)
    )
    assert metrics.response_count == 1
    assert metrics.concise_chunking_activation_count == 1
    assert metrics.chunk_count == len(chunks)
    assert metrics.max_chunk_chars == max(len(chunk) for chunk in chunks)
    assert metrics.average_chunk_chars > 0.0
    assert metrics.buffer_flush_count >= 1


@pytest.mark.asyncio
async def test_concise_voice_policy_processor_splits_chinese_chunks():
    policy = _concise_policy(max_chars=28)
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(policy, metrics_recorder=recorder)
    text = "先确认最小复现路径，然后只改变一个变量。观察结果后，再决定下一步。"

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    chunks = [frame.text for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    metrics = recorder.snapshot()
    assert "".join(chunks) == text
    assert len(chunks) >= 2
    assert all(len(chunk) <= policy.max_spoken_chunk_chars for chunk in chunks)
    assert metrics.chunk_count == len(chunks)
    assert metrics.max_chunk_chars <= policy.max_spoken_chunk_chars


@pytest.mark.asyncio
async def test_voice_policy_processor_respects_chunk_boundary_capability_noop():
    policy = _concise_policy(max_chars=24)
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(
        policy,
        capabilities=BrainVoiceBackendCapabilities(
            backend_label="test-no-chunk",
            supports_chunk_boundaries=False,
            supports_interruption_flush=True,
            supports_interruption_discard=True,
            reason_codes=("test_no_chunk_boundaries",),
        ),
        metrics_recorder=recorder,
    )
    text_frame = LLMTextFrame(
        "This text would be chunked if the backend supported chunk boundaries."
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(text_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    metrics = recorder.snapshot()

    assert text_frame in [frame for frame, _direction in processor.pushed]
    assert not any(isinstance(frame, AggregatedTextFrame) for frame, _direction in processor.pushed)
    assert metrics.response_count == 1
    assert metrics.concise_chunking_activation_count == 0
    assert metrics.last_chunking_mode == "off"


@pytest.mark.asyncio
async def test_voice_policy_processor_does_not_rewrite_skip_or_transcription_frames():
    policy = _concise_policy(max_chars=24)
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(policy, metrics_recorder=recorder)
    skipped = LLMTextFrame("This should pass through unchanged.")
    skipped.skip_tts = True
    interim = InterimTranscriptionFrame("hello", user_id="u1", timestamp="t1")
    final = TranscriptionFrame("hello", user_id="u1", timestamp="t2")

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(skipped, FrameDirection.DOWNSTREAM)
    await processor.process_frame(interim, FrameDirection.DOWNSTREAM)
    await processor.process_frame(final, FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    pushed = [frame for frame, _direction in processor.pushed]
    assert skipped in pushed
    assert interim in pushed
    assert final in pushed
    assert not any(isinstance(frame, AggregatedTextFrame) for frame in pushed)
    assert recorder.snapshot().chunk_count == 0


@pytest.mark.asyncio
async def test_voice_policy_processor_flushes_or_discards_on_turn_boundaries():
    policy = _concise_policy(max_chars=80)
    flush_recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(policy, metrics_recorder=flush_recorder)

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        LLMTextFrame("Buffered until response end"), FrameDirection.DOWNSTREAM
    )
    assert not any(isinstance(frame, AggregatedTextFrame) for frame, _ in processor.pushed)

    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert any(
        isinstance(frame, AggregatedTextFrame) and frame.text == "Buffered until response end"
        for frame, _ in processor.pushed
    )
    flush_metrics = flush_recorder.snapshot()
    assert flush_metrics.chunk_count == 1
    assert flush_metrics.buffer_flush_count == 1

    discard_recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(
        policy,
        capabilities=BrainVoiceBackendCapabilities(
            backend_label="test-discard",
            supports_interruption_discard=True,
            reason_codes=("test_interruption_discard_supported",),
        ),
        metrics_recorder=discard_recorder,
    )
    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        LLMTextFrame("discard this buffered text"), FrameDirection.DOWNSTREAM
    )
    await processor.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    assert not any(
        isinstance(frame, AggregatedTextFrame) and frame.text == "discard this buffered text"
        for frame, _ in processor.pushed
    )
    discard_metrics = discard_recorder.snapshot()
    assert discard_metrics.interruption_frame_count == 1
    assert discard_metrics.buffer_discard_count == 1
    assert discard_metrics.chunk_count == 0


@pytest.mark.asyncio
async def test_provider_neutral_interruption_flushes_buffered_sentence_instead_of_dropping():
    policy = _concise_policy(max_chars=80)
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(policy, metrics_recorder=recorder)

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        LLMTextFrame("This unfinished sentence should still reach TTS"),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    chunks = [frame.text for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    metrics = recorder.snapshot()

    assert chunks == ["This unfinished sentence should still reach TTS"]
    assert metrics.interruption_frame_count == 1
    assert metrics.buffer_flush_count == 1
    assert metrics.buffer_discard_count == 0


@pytest.mark.asyncio
async def test_local_http_wav_voice_policy_buffers_short_turn_until_response_end():
    policy = _concise_policy(max_chars=48)
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(
        policy,
        tts_backend="local-http-wav",
        metrics_recorder=recorder,
    )
    text = "第一句已经完整。第二句也完整。这样给梅洛一次合成，避免每句话都重新淡出。"

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text[:18]), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text[18:]), FrameDirection.DOWNSTREAM)

    assert not any(isinstance(frame, AggregatedTextFrame) for frame, _ in processor.pushed)

    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)
    chunks = [frame.text for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    metrics = recorder.snapshot()

    assert chunks == [text]
    assert metrics.chunk_count == 1
    assert metrics.average_chunks_per_response == 1.0


@pytest.mark.asyncio
async def test_local_http_wav_voice_policy_uses_large_bounded_chunks_for_long_turns():
    policy = _concise_policy(max_chars=48)
    recorder = BrainExpressionVoiceMetricsRecorder()
    processor = CaptureVoicePolicyProcessor(
        policy,
        tts_backend="local-http-wav",
        metrics_recorder=recorder,
    )
    sentence = "先确认输入，再确认模型，再确认语音输出。"
    text = sentence * 32

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMTextFrame(text), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    chunks = [frame.text for frame, _ in processor.pushed if isinstance(frame, AggregatedTextFrame)]
    metrics = recorder.snapshot()

    assert "".join(chunks) == text
    assert 1 < len(chunks) < 12
    assert all(len(chunk) <= 220 for chunk in chunks)
    assert metrics.chunk_count == len(chunks)

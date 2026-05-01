"""Blink brain frame processors."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Mapping

from blink.brain.actions import (
    EmbodiedCapabilityDispatcher,
    EmbodiedCommandInterpreter,
    capability_id_for_action,
)
from blink.brain.autonomy import BrainReevaluationConditionKind, BrainReevaluationTrigger
from blink.brain.context.compiler import BrainContextCompiler
from blink.brain.events import BrainEventType
from blink.brain.executive import BrainExecutive
from blink.brain.identity import BRAIN_CONTEXT_HEADER
from blink.brain.memory import (
    BrainMemoryConsolidator,
    extract_memory_candidates,
    extract_task_candidates,
)
from blink.brain.persona import (
    BrainExpressionVoiceMetricsRecorder,
    BrainExpressionVoicePolicy,
    BrainPersonaModality,
    BrainRealtimeVoiceActuationPlan,
    BrainVoiceBackendCapabilities,
    compile_realtime_voice_actuation_plan,
)
from blink.brain.speech_director import (
    SpeechChunkBudgetV3,
    SpeechDirectorMode,
    SpeechPerformanceChunk,
    build_speech_chunk_frame_metadata,
    next_kokoro_speech_chunk,
    next_melo_speech_chunk,
)
from blink.brain.store import BrainStore
from blink.frames.frames import (
    AggregatedTextFrame,
    AggregationType,
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextAssistantTimestampFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.transcriptions.language import Language

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from blink.brain.context.compiler import BrainCompiledContextPacket
    from blink.brain.memory_v2 import BrainMemoryUseTrace
    from blink.interaction.actor_control_frame_v3 import ActorControlScheduler
    from blink.processors.aggregators.llm_context import LLMContext


def latest_user_text_from_context(context: LLMContext) -> str:
    """Return the latest plain-text user message in a context."""
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue

        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if text:
                        parts.append(text)
            if parts:
                return " ".join(parts)

    return ""


def latest_assistant_text_from_context(context: LLMContext) -> str:
    """Return the latest assistant text in a context."""
    for message in reversed(context.get_messages()):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


def _maybe_json_load(value):
    """Best-effort JSON decode for context payload fragments."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def latest_turn_tool_calls_from_context(context: LLMContext) -> list[dict]:
    """Return structured tool calls associated with the latest user turn."""
    messages = list(context.get_messages())
    last_user_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            last_user_index = index

    relevant_messages = messages[last_user_index + 1 :] if last_user_index >= 0 else messages
    records: list[dict] = []
    records_by_id: dict[str, dict] = {}

    def ensure_record(
        tool_call_id: str | None, *, function_name=None, arguments=None
    ) -> dict | None:
        if not tool_call_id:
            return None
        record = records_by_id.get(tool_call_id)
        if record is None:
            record = {
                "tool_call_id": tool_call_id,
                "function_name": function_name,
                "arguments": arguments,
                "result": None,
            }
            records_by_id[tool_call_id] = record
            records.append(record)
        else:
            if function_name is not None and not record.get("function_name"):
                record["function_name"] = function_name
            if arguments is not None and record.get("arguments") in (None, "", {}):
                record["arguments"] = arguments
        return record

    for message in relevant_messages:
        if not isinstance(message, dict):
            continue

        if message.get("role") == "assistant" and isinstance(message.get("tool_calls"), list):
            for tool_call in message.get("tool_calls", []):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function", {})
                arguments = function.get("arguments") if isinstance(function, dict) else None
                ensure_record(
                    str(tool_call.get("id", "")).strip() or None,
                    function_name=function.get("name") if isinstance(function, dict) else None,
                    arguments=_maybe_json_load(arguments),
                )
            continue

        if message.get("type") == "function_call":
            ensure_record(
                str(message.get("call_id") or message.get("id") or "").strip() or None,
                function_name=message.get("name"),
                arguments=_maybe_json_load(message.get("arguments")),
            )
            continue

        if message.get("role") == "tool":
            record = ensure_record(str(message.get("tool_call_id", "")).strip() or None)
            if record is not None:
                record["result"] = _maybe_json_load(message.get("content"))
            continue

        if message.get("role") == "developer" and isinstance(message.get("content"), str):
            payload = _maybe_json_load(message.get("content"))
            if isinstance(payload, dict) and payload.get("type") == "async_tool":
                record = ensure_record(
                    str(payload.get("tool_call_id", "")).strip() or None,
                )
                if record is not None:
                    record["async_status"] = payload.get("status")
                    if "result" in payload:
                        record["result"] = _maybe_json_load(payload.get("result"))

    return records


_VOICE_CHUNK_BOUNDARIES = ".!?。！？；;\n"
_LOCAL_HTTP_WAV_BACKENDS = {"local-http-wav", "local_http_wav"}
_LOCAL_HTTP_WAV_TURN_CHUNK_LIMIT = 220
_LOCAL_HTTP_WAV_OVERFLOW_FLUSH_CHARS = _LOCAL_HTTP_WAV_TURN_CHUNK_LIMIT * 2
_SPEECH_DIRECTOR_BACKLOG_WARN_CHUNKS = 6

BrainPerformanceEmit = Callable[..., object]


class BrainVoiceInputHealthProcessor(FrameProcessor):
    """Observe browser voice input and STT progress without mutating frames."""

    def __init__(self, *, runtime: object, phase: str = "post_stt"):
        """Initialize the processor."""
        normalized_phase = phase if phase in {"pre_stt", "post_stt"} else "post_stt"
        super().__init__(name=f"brain-voice-input-health-{normalized_phase}")
        self._runtime = runtime
        self.phase = normalized_phase

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Record public-safe input/STT state and pass frames through unchanged."""
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            self._record_frame(frame)

        await self.push_frame(frame, direction)

    def _record_frame(self, frame: Frame):
        if self.phase == "pre_stt":
            if isinstance(frame, AudioRawFrame):
                recorder = getattr(self._runtime, "note_voice_input_audio_frame", None)
                if callable(recorder):
                    recorder()
                return
            if isinstance(frame, VADUserStartedSpeakingFrame):
                recorder = getattr(self._runtime, "note_voice_input_speech_started", None)
                if callable(recorder):
                    recorder()
                return
            if isinstance(frame, VADUserStoppedSpeakingFrame):
                recorder = getattr(self._runtime, "note_voice_input_speech_stopped", None)
                if callable(recorder):
                    recorder()
                return
            return

        if isinstance(frame, InterimTranscriptionFrame):
            recorder = getattr(self._runtime, "note_voice_input_interim_transcription", None)
            if callable(recorder):
                recorder(frame.text)
            return
        if isinstance(frame, TranscriptionFrame):
            recorder = getattr(self._runtime, "note_voice_input_transcription", None)
            if callable(recorder):
                recorder(frame.text)
            return
        if isinstance(frame, ErrorFrame):
            recorder = getattr(self._runtime, "note_voice_input_stt_error", None)
            if callable(recorder):
                recorder(type(frame).__name__)


class BrainExpressionVoicePolicyProcessor(FrameProcessor):
    """Apply provider-neutral expression voice policy before TTS."""

    def __init__(
        self,
        *,
        policy_provider: Callable[[], BrainExpressionVoicePolicy],
        actuation_plan_provider: Callable[[], BrainRealtimeVoiceActuationPlan] | None = None,
        capabilities_provider: Callable[[], BrainVoiceBackendCapabilities] | None = None,
        tts_backend: str | None = None,
        language: str | None = None,
        metrics_recorder: BrainExpressionVoiceMetricsRecorder | None = None,
        speech_director_mode: SpeechDirectorMode | str = "unavailable",
        performance_emit: BrainPerformanceEmit | None = None,
        actor_control_scheduler: "ActorControlScheduler | None" = None,
        performance_plan_provider: Callable[[], object] | None = None,
        speech_queue_controller: object | None = None,
        enable_direct_mode: bool = False,
    ):
        """Initialize the processor."""
        super().__init__(
            name="brain-expression-voice-policy",
            enable_direct_mode=enable_direct_mode,
        )
        self._policy_provider = policy_provider
        self._actuation_plan_provider = actuation_plan_provider
        self._capabilities_provider = capabilities_provider
        self._tts_backend = tts_backend
        self._language = str(language or "").strip()
        self._metrics_recorder = metrics_recorder
        self._speech_director_mode = str(speech_director_mode or "unavailable")
        self._performance_emit = performance_emit
        self._actor_control_scheduler = actor_control_scheduler
        self._performance_plan_provider = performance_plan_provider
        self._speech_queue_controller = speech_queue_controller
        self._policy: BrainExpressionVoicePolicy | None = None
        self._actuation_plan: BrainRealtimeVoiceActuationPlan | None = None
        self._performance_plan_v3: dict[str, Any] | None = None
        self._buffer = ""
        self._generation_id = ""
        self._turn_id = ""
        self._turn_index = 0
        self._chunk_index = 0
        self._response_started_at: float | None = None
        self._first_subtitle_emitted = False
        self._pending_end_frame: LLMFullResponseEndFrame | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Chunk LLM text when concise voice policy is active."""
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._policy = self._resolve_policy()
            self._actuation_plan = self._resolve_actuation_plan()
            self._performance_plan_v3 = self._resolve_performance_plan()
            self._buffer = ""
            self._turn_index += 1
            self._chunk_index = 0
            self._generation_id = f"speech-{uuid.uuid4().hex[:12]}"
            self._turn_id = f"turn-{self._turn_index}"
            self._response_started_at = time.monotonic()
            self._first_subtitle_emitted = False
            self._pending_end_frame = None
            if self._metrics_recorder is not None:
                self._metrics_recorder.record_response_start(
                    self._actuation_plan,
                    speech_director_mode=self._effective_speech_director_mode(),
                )
            self._emit_speech_event(
                event_type="speech.generation_start",
                metadata={
                    "generation_id": self._generation_id,
                    "turn_id": self._turn_id,
                    "director_mode": self._effective_speech_director_mode(),
                },
                reason_codes=("speech:generation_start",),
            )
            start_generation = getattr(self._speech_queue_controller, "start_generation", None)
            if callable(start_generation):
                start_generation(generation_id=self._generation_id, turn_id=self._turn_id)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            discarded_buffer = bool(self._buffer.strip() and self._speech_director_active())
            if not discarded_buffer:
                discarded_buffer = bool(
                    self._buffer.strip()
                    and self._actuation_plan is not None
                    and self._actuation_plan.interruption_discard_enabled
                )
            if (
                self._buffer.strip()
                and not discarded_buffer
                and self._actuation_plan is not None
                and self._actuation_plan.interruption_flush_enabled
            ):
                await self._flush_chunks(direction=direction, force=True)
            if self._metrics_recorder is not None:
                self._metrics_recorder.record_interruption(
                    discarded_buffer=discarded_buffer,
                    accepted=bool(self._actuation_plan and self._actuation_plan.available),
                )
            self._buffer = ""
            self._pending_end_frame = None
            self._performance_plan_v3 = None
            note_interruption = getattr(self._speech_queue_controller, "note_interruption", None)
            if callable(note_interruption):
                note_interruption(dropped_count=1 if discarded_buffer else 0)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            self._pending_end_frame = frame
            if self._actuation_active():
                await self._flush_chunks(direction=direction, force=True)
            if self._buffer.strip():
                return
            await self._push_pending_end_frame(direction=direction)
            return

        if not self._actuation_active() or not _voice_policy_text_frame(frame):
            await self.push_frame(frame, direction)
            return

        self._buffer += frame.text
        if self._hold_for_turn_level_synthesis() and not self._use_melo_speech_director():
            if len(self._buffer) >= _LOCAL_HTTP_WAV_OVERFLOW_FLUSH_CHARS:
                await self._flush_chunks(direction=direction, force=False)
            return
        await self._flush_chunks(direction=direction, force=False)

    def _resolve_policy(self) -> BrainExpressionVoicePolicy | None:
        try:
            return self._policy_provider()
        except Exception:
            return None

    def _resolve_actuation_plan(self) -> BrainRealtimeVoiceActuationPlan | None:
        try:
            if self._actuation_plan_provider is not None:
                return self._actuation_plan_provider()
            capabilities = self._capabilities_provider() if self._capabilities_provider else None
            return compile_realtime_voice_actuation_plan(
                self._policy,
                capabilities=capabilities,
                tts_backend=self._tts_backend,
            )
        except Exception:
            return None

    def _resolve_performance_plan(self) -> dict[str, Any] | None:
        if self._performance_plan_provider is None:
            return None
        try:
            plan = self._performance_plan_provider()
        except Exception:
            return None
        as_dict = getattr(plan, "as_dict", None)
        if callable(as_dict):
            payload = as_dict()
            return dict(payload) if isinstance(payload, Mapping) else None
        return dict(plan) if isinstance(plan, Mapping) else None

    def _actuation_active(self) -> bool:
        if self._use_melo_speech_director() or self._use_kokoro_speech_director():
            return bool(self._actuation_plan and self._actuation_plan.available)
        return bool(
            self._actuation_plan
            and self._actuation_plan.available
            and self._actuation_plan.chunk_boundaries_enabled
            and self._actuation_plan.max_spoken_chunk_chars > 0
        )

    async def _flush_chunks(
        self,
        *,
        direction: FrameDirection,
        force: bool,
        max_chunks_per_flush_override: int | None = None,
    ):
        if self._actuation_plan is None:
            return
        budget = self._chunk_budget_v3()
        limit = budget.hard_max_chars
        min_chars = 0 if force else budget.min_chars
        max_chunks_per_flush = (
            max(1, int(max_chunks_per_flush_override))
            if max_chunks_per_flush_override is not None
            else budget.max_chunks_per_flush
        )
        emitted_chunks = 0
        while self._buffer.strip():
            if max_chunks_per_flush > 0 and emitted_chunks >= max_chunks_per_flush:
                self._emit_speech_event(
                    event_type="speech.plan_chunk_budget_held",
                    metadata={
                        "generation_id": self._generation_id or "speech-unavailable",
                        "turn_id": self._turn_id or "turn-unavailable",
                        "held_chunk_count": 1,
                        "remaining_chars": len(self._buffer),
                        "max_chunks_per_flush": max_chunks_per_flush,
                        "director_mode": self._effective_speech_director_mode(),
                    },
                    reason_codes=("speech:plan_chunk_budget_held",),
                )
                break
            chunk, remaining, pause_after_ms, reason_codes = self._next_chunk(
                limit=limit,
                force=force,
                min_chars=min_chars,
            )
            if chunk is None:
                break
            if self._speech_director_active() and not self._speech_lookahead_can_emit():
                note_held = getattr(self._speech_queue_controller, "note_lookahead_held", None)
                if callable(note_held):
                    note_held(count=1)
                self._emit_speech_event(
                    event_type="speech.lookahead_held",
                    metadata={
                        "generation_id": self._generation_id or "speech-unavailable",
                        "turn_id": self._turn_id or "turn-unavailable",
                        "held_chunk_count": 1,
                        "remaining_chars": len(self._buffer),
                        "speech_queue_depth": self._speech_lookahead_depth(),
                        "director_mode": self._effective_speech_director_mode(),
                    },
                    reason_codes=("speech:lookahead_held",),
                )
                break
            self._buffer = remaining
            emitted_chunks += 1
            queue_depth = emitted_chunks + (1 if self._buffer.strip() else 0)
            if self._metrics_recorder is not None:
                self._metrics_recorder.record_chunk(chunk)
                self._metrics_recorder.record_speech_queue_depth(queue_depth)
                if (
                    self._speech_director_active()
                    and not self._first_subtitle_emitted
                    and self._response_started_at is not None
                ):
                    self._metrics_recorder.record_first_subtitle_latency(
                        (time.monotonic() - self._response_started_at) * 1000.0
                    )
            frame = AggregatedTextFrame(chunk, AggregationType.SENTENCE)
            if self._speech_director_active():
                speech_chunk = self._build_speech_chunk(
                    text=chunk,
                    pause_after_ms=pause_after_ms,
                    reason_codes=(*budget.reason_codes, *reason_codes),
                )
                if not self._first_subtitle_emitted:
                    self._first_subtitle_emitted = True
                self._emit_speech_event(
                    event_type="speech.subtitle_ready",
                    metadata={
                        **speech_chunk.public_metadata(queue_depth=queue_depth),
                        "director_mode": self._effective_speech_director_mode(),
                    },
                    reason_codes=("speech:subtitle_ready", *speech_chunk.reason_codes),
                )
                note_subtitle = getattr(self._speech_queue_controller, "note_subtitle_ready", None)
                if callable(note_subtitle):
                    note_subtitle()
                build_speech_chunk_frame_metadata(frame, speech_chunk)
            await self.push_frame(
                frame,
                direction,
            )
            if emitted_chunks % _SPEECH_DIRECTOR_BACKLOG_WARN_CHUNKS == 0 and self._buffer.strip():
                self._emit_speech_event(
                    event_type="speech.backlog_degraded",
                    metadata={
                        "generation_id": self._generation_id or "speech-unavailable",
                        "turn_id": self._turn_id or "turn-unavailable",
                        "emitted_chunk_count": emitted_chunks,
                        "remaining_chars": len(self._buffer),
                        "max_chunks_per_flush": _SPEECH_DIRECTOR_BACKLOG_WARN_CHUNKS,
                        "director_mode": self._effective_speech_director_mode(),
                    },
                    reason_codes=("speech:backlog_degraded",),
                )
                await asyncio.sleep(0)
        if self._metrics_recorder is not None:
            self._metrics_recorder.record_buffer_flush(emitted_chunk_count=emitted_chunks)
        if not self._buffer.strip():
            await self._push_pending_end_frame(direction=direction)

    async def drain_held_speech_chunks(self) -> None:
        """Drain one held speech chunk after a downstream TTS queue boundary."""
        self._performance_plan_v3 = self._resolve_performance_plan() or self._performance_plan_v3
        if not self._buffer.strip() or not self._actuation_active():
            if not self._buffer.strip():
                await self._push_pending_end_frame(direction=FrameDirection.DOWNSTREAM)
            return
        if self._speech_director_active() and not self._speech_lookahead_can_emit():
            return
        await self._flush_chunks(
            direction=FrameDirection.DOWNSTREAM,
            force=self._pending_end_frame is not None,
            max_chunks_per_flush_override=1,
        )

    def _speech_lookahead_can_emit(self) -> bool:
        can_emit_session = getattr(self._speech_queue_controller, "can_emit", None)
        if callable(can_emit_session):
            try:
                return bool(can_emit_session())
            except Exception:
                return True
        if self._actor_control_scheduler is None:
            return True
        can_emit = getattr(self._actor_control_scheduler, "speech_lookahead_can_emit", None)
        if not callable(can_emit):
            return True
        try:
            return bool(can_emit())
        except Exception:
            return True

    def _speech_lookahead_depth(self) -> int:
        speech_state = getattr(self._speech_queue_controller, "as_dict", None)
        if callable(speech_state):
            try:
                payload = speech_state()
                if isinstance(payload, dict):
                    return max(0, int(payload.get("speech_chunks_outstanding") or 0))
            except Exception:
                return 0
        scheduler = self._actor_control_scheduler
        state = getattr(scheduler, "_state", None)
        if state is None:
            return 0
        return max(0, int(getattr(state, "outstanding_speech_chunks", 0)))

    async def _push_pending_end_frame(self, *, direction: FrameDirection) -> None:
        if self._pending_end_frame is None or self._buffer.strip():
            return
        frame = self._pending_end_frame
        self._pending_end_frame = None
        self._policy = None
        self._actuation_plan = None
        self._performance_plan_v3 = None
        await self.push_frame(frame, direction)

    def _chunk_budget_v3(self) -> SpeechChunkBudgetV3:
        if self._actuation_plan is None:
            return SpeechChunkBudgetV3.from_plan(
                director_mode=self._effective_speech_director_mode(),
                tts_backend=self._tts_backend,
                actuation_chunk_limit=1,
            )
        return SpeechChunkBudgetV3.from_plan(
            director_mode=self._effective_speech_director_mode(),
            tts_backend=self._tts_backend,
            actuation_chunk_limit=max(1, int(self._actuation_plan.max_spoken_chunk_chars)),
            plan_budget=self._plan_budget_payload(),
            local_http_wav_turn_chunk_limit=_LOCAL_HTTP_WAV_TURN_CHUNK_LIMIT,
        )

    def _chunk_target(self, *, limit: int) -> int:
        budget = self._chunk_budget_v3()
        return max(1, min(limit, budget.target_chars))

    def _max_chunks_per_flush(self) -> int:
        return self._chunk_budget_v3().max_chunks_per_flush

    def _plan_budget_payload(self) -> Mapping[str, Any] | None:
        plan = self._performance_plan_v3 if isinstance(self._performance_plan_v3, dict) else {}
        budget = plan.get("speech_chunk_budget")
        if not isinstance(budget, Mapping):
            return None
        return budget

    def _hold_for_turn_level_synthesis(self) -> bool:
        return self._is_local_http_wav_backend()

    def _is_local_http_wav_backend(self) -> bool:
        return str(self._tts_backend or "").strip().lower() in _LOCAL_HTTP_WAV_BACKENDS

    def _use_melo_speech_director(self) -> bool:
        return self._effective_speech_director_mode() == "melo_chunked" and (
            self._is_local_http_wav_backend()
        )

    def _use_kokoro_speech_director(self) -> bool:
        return (
            self._effective_speech_director_mode() == "kokoro_chunked"
            and str(self._tts_backend or "").strip().lower() == "kokoro"
        )

    def _speech_director_active(self) -> bool:
        return self._effective_speech_director_mode() != "unavailable"

    def _effective_speech_director_mode(self) -> str:
        mode = str(self._speech_director_mode or "unavailable")
        if mode == "kokoro_passthrough":
            return "kokoro_chunked"
        if mode in {"melo_chunked", "kokoro_chunked"}:
            return mode
        return "unavailable"

    def _next_chunk(
        self,
        *,
        limit: int,
        force: bool,
        min_chars: int,
    ) -> tuple[str, str, int, tuple[str, ...]] | tuple[None, str, int, tuple[str, ...]]:
        if self._use_melo_speech_director():
            return next_melo_speech_chunk(
                self._buffer,
                force=force,
                min_chars=min_chars,
                target_chars=self._chunk_target(limit=limit),
                hard_max_chars=limit,
            )
        if self._use_kokoro_speech_director():
            return next_kokoro_speech_chunk(
                self._buffer,
                force=force,
                min_chars=min_chars,
                target_chars=self._chunk_target(limit=limit),
                hard_max_chars=limit,
            )
        chunk, remaining = _next_voice_chunk(
            self._buffer,
            limit=limit,
            force=force,
            min_chars=min_chars,
        )
        if chunk is None:
            return None, remaining, 0, ("speech_director:waiting",)
        return chunk, remaining, 0, ("speech_director:legacy_boundary",)

    def _build_speech_chunk(
        self,
        *,
        text: str,
        pause_after_ms: int,
        reason_codes: tuple[str, ...],
    ) -> SpeechPerformanceChunk:
        self._chunk_index += 1
        return SpeechPerformanceChunk(
            role="assistant",
            text=str(text or "").strip(),
            language=self._speech_language(),
            tts_backend=str(self._tts_backend or "unknown").strip() or "unknown",
            display_text=str(text or "").strip(),
            interruptible=True,
            pause_after_ms=max(0, int(pause_after_ms)),
            generation_token=self._generation_id or "speech-unavailable",
            turn_id=self._turn_id or "turn-unavailable",
            chunk_index=self._chunk_index,
            reason_codes=tuple(reason_codes),
        )

    def _speech_language(self) -> str:
        if self._language:
            return self._language
        backend = str(self._tts_backend or "").strip().lower()
        if backend in _LOCAL_HTTP_WAV_BACKENDS:
            return "zh"
        if backend == "kokoro":
            return "en"
        return "unknown"

    def _emit_speech_event(
        self,
        *,
        event_type: str,
        metadata: dict[str, object],
        reason_codes: tuple[str, ...],
    ) -> None:
        if self._performance_emit is None or self._effective_speech_director_mode() == "unavailable":
            return
        self._performance_emit(
            event_type=event_type,
            source="speech_director",
            mode="thinking",
            metadata=metadata,
            reason_codes=reason_codes,
        )


def _voice_policy_text_frame(frame: Frame) -> bool:
    return (
        isinstance(frame, TextFrame)
        and not isinstance(
            frame,
            (
                AggregatedTextFrame,
                InterimTranscriptionFrame,
                TranscriptionFrame,
            ),
        )
        and not bool(getattr(frame, "skip_tts", False))
    )


def _next_voice_chunk(
    text: str,
    *,
    limit: int,
    force: bool,
    min_chars: int = 0,
) -> tuple[str, str] | tuple[None, str]:
    normalized = str(text or "")
    if not normalized.strip():
        return None, ""
    safe_min_chars = 0 if force else max(0, min(int(min_chars), int(limit)))
    if len(normalized) <= limit and not force:
        boundary = _last_boundary_before(normalized, limit=len(normalized))
        if boundary is None:
            return None, normalized
        if len(normalized[:boundary].strip()) < safe_min_chars:
            return None, normalized
        return _split_at(normalized, boundary)
    if len(normalized) <= limit:
        return normalized.strip(), ""

    boundary = _last_boundary_before(normalized, limit=limit)
    if boundary is None:
        boundary = _last_space_before(normalized, limit=limit)
    if boundary is None or boundary < max(24, limit // 3, safe_min_chars):
        boundary = limit
    return _split_at(normalized, boundary)


def _last_boundary_before(text: str, *, limit: int) -> int | None:
    bounded = text[:limit]
    indexes = [bounded.rfind(marker) for marker in _VOICE_CHUNK_BOUNDARIES]
    index = max(indexes)
    return index + 1 if index >= 0 else None


def _last_space_before(text: str, *, limit: int) -> int | None:
    index = text[:limit].rfind(" ")
    return index + 1 if index >= 0 else None


def _split_at(text: str, index: int) -> tuple[str, str]:
    chunk = text[:index].strip()
    remaining = text[index:].lstrip()
    return chunk, remaining


class _BrainTurnPersistenceHelper:
    """Persist one completed assistant turn from the shared context."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
        context: LLMContext,
        consolidator: BrainMemoryConsolidator,
        executive: BrainExecutive | None = None,
        memory_use_trace_resolver: Callable[[], "BrainMemoryUseTrace | None"] | None = None,
        memory_use_trace_committed_callback: Callable[["BrainMemoryUseTrace"], None] | None = None,
    ):
        self._store = store
        self._session_resolver = session_resolver
        self._context = context
        self._consolidator = consolidator
        self._executive = executive
        self._memory_use_trace_resolver = memory_use_trace_resolver
        self._memory_use_trace_committed_callback = memory_use_trace_committed_callback
        self._last_recorded_signature: tuple[str, str, str, int] | None = None

    async def persist_completed_turn(self, *, source: str):
        """Persist the latest assistant turn when the context is complete and new."""
        user_text = latest_user_text_from_context(self._context)
        assistant_text = latest_assistant_text_from_context(self._context)
        tool_calls = latest_turn_tool_calls_from_context(self._context)
        tool_signature = json.dumps(tool_calls, ensure_ascii=False, sort_keys=True)
        signature = (user_text, assistant_text, tool_signature, len(self._context.get_messages()))
        if not user_text or not assistant_text or signature == self._last_recorded_signature:
            return

        session_ids = self._session_resolver()
        parent_event = self._store.latest_brain_event(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
            event_types=(BrainEventType.USER_TURN_TRANSCRIBED,),
        )
        for tool_call in tool_calls:
            tool_call_id = str(tool_call.get("tool_call_id", "")).strip() or None
            self._store.append_brain_event(
                event_type=BrainEventType.TOOL_CALLED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source=source,
                correlation_id=tool_call_id,
                causal_parent_id=parent_event.event_id if parent_event is not None else None,
                payload={
                    "tool_call_id": tool_call_id,
                    "function_name": tool_call.get("function_name"),
                    "arguments": tool_call.get("arguments"),
                },
            )
            if tool_call.get("result") is not None:
                self._store.append_brain_event(
                    event_type=BrainEventType.TOOL_COMPLETED,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source=source,
                    correlation_id=tool_call_id,
                    causal_parent_id=parent_event.event_id if parent_event is not None else None,
                    payload={
                        "tool_call_id": tool_call_id,
                        "function_name": tool_call.get("function_name"),
                        "result": tool_call.get("result"),
                    },
                )
        assistant_event = self._store.append_brain_event(
            event_type=BrainEventType.ASSISTANT_TURN_ENDED,
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            source=source,
            causal_parent_id=parent_event.event_id if parent_event is not None else None,
            payload={"text": assistant_text},
        )
        if self._memory_use_trace_resolver is not None:
            trace = self._memory_use_trace_resolver()
            if trace is not None and trace.refs:
                persisted_trace = self._store.append_memory_use_trace(
                    trace=trace,
                    session_id=session_ids.session_id,
                    source=source,
                    causal_parent_id=assistant_event.event_id,
                    ts=assistant_event.ts,
                )
                if self._memory_use_trace_committed_callback is not None:
                    self._memory_use_trace_committed_callback(persisted_trace)
        self._store.add_episode(
            agent_id=session_ids.agent_id,
            user_id=session_ids.user_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
            user_text=user_text,
            assistant_text=assistant_text,
            assistant_summary=assistant_text[:280],
            tool_calls=tool_calls,
        )
        self._consolidator.run_once(
            user_id=session_ids.user_id,
            thread_id=session_ids.thread_id,
        )
        if self._executive is not None:
            await self._executive.run_turn_end_pass()
        self._last_recorded_signature = signature


def _record_user_turn_text(
    *,
    store: BrainStore,
    session_resolver,
    language: Language,
    latest_user_text: str,
    source: str,
):
    """Persist one user text turn plus hot-path typed memory extraction."""
    session_ids = session_resolver()
    store.ensure_user(
        user_id=session_ids.user_id,
        language=language.value,
    )
    user_turn_event = store.append_brain_event(
        event_type=BrainEventType.USER_TURN_TRANSCRIBED,
        agent_id=session_ids.agent_id,
        user_id=session_ids.user_id,
        session_id=session_ids.session_id,
        thread_id=session_ids.thread_id,
        source=source,
        payload={"text": latest_user_text},
    )
    for candidate in extract_memory_candidates(latest_user_text):
        store.remember_fact(
            user_id=session_ids.user_id,
            namespace=candidate.namespace,
            subject=candidate.subject,
            value=candidate.value,
            rendered_text=candidate.rendered_text,
            confidence=candidate.confidence,
            singleton=candidate.singleton,
            source_event_id=user_turn_event.event_id,
            provenance={
                "source": "hot_path",
                "source_event_id": user_turn_event.event_id,
                "source_event_type": user_turn_event.event_type,
            },
            source_episode_id=None,
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
            thread_id=session_ids.thread_id,
        )
    for candidate in extract_task_candidates(latest_user_text):
        store.upsert_task(
            user_id=session_ids.user_id,
            title=candidate.title,
            details=candidate.details,
            status=candidate.status,
            thread_id=session_ids.thread_id,
            source_event_id=user_turn_event.event_id,
            provenance={
                "source": "hot_path",
                "source_event_id": user_turn_event.event_id,
                "source_event_type": user_turn_event.event_type,
            },
            agent_id=session_ids.agent_id,
            session_id=session_ids.session_id,
        )
    return user_turn_event


class BrainEventRecorderProcessor(FrameProcessor):
    """Record typed turn-boundary events on the frame pipeline hot path."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
        executive: BrainExecutive | None = None,
    ):
        """Initialize the event recorder.

        Args:
            store: Canonical local-first brain store.
            session_resolver: Callable returning stable runtime session ids.
            executive: Optional executive reevaluated on user turn close.
        """
        super().__init__(name="brain-event-recorder")
        self._store = store
        self._session_resolver = session_resolver
        self._executive = executive

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Append typed turn-boundary events without mutating frame flow."""
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM:
            session_ids = None
            event_type = None
            if isinstance(frame, UserStartedSpeakingFrame):
                event_type = BrainEventType.USER_TURN_STARTED
            elif isinstance(frame, UserStoppedSpeakingFrame):
                event_type = BrainEventType.USER_TURN_ENDED
            elif isinstance(frame, BotStartedSpeakingFrame):
                event_type = BrainEventType.ASSISTANT_TURN_STARTED

            if event_type is not None:
                try:
                    session_ids = self._session_resolver()
                    event = self._store.append_brain_event(
                        event_type=event_type,
                        agent_id=session_ids.agent_id,
                        user_id=session_ids.user_id,
                        session_id=session_ids.session_id,
                        thread_id=session_ids.thread_id,
                        source="pipeline",
                        payload={},
                    )
                    if (
                        self._executive is not None
                        and event_type == BrainEventType.USER_TURN_ENDED
                    ):
                        self._executive.run_presence_director_reevaluation(
                            BrainReevaluationTrigger(
                                kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
                                summary="Reevaluate held candidates after the user turn closes.",
                                details={"turn": "user"},
                                source_event_type=event.event_type,
                                source_event_id=event.event_id,
                                ts=event.ts,
                            )
                        )
                        await self._executive.run_commitment_wake_router(
                            boundary_kind="user_turn_end",
                            source_event=event,
                        )
                except Exception as exc:
                    logger.warning(
                        "Brain event recorder persistence failed; continuing without frame error: %s",
                        exc,
                    )

        await self.push_frame(frame, direction)


class HotPathMemoryExtractor(FrameProcessor):
    """Persist typed semantic memory on the conversation hot path."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
        language: Language,
    ):
        """Initialize the extractor.

        Args:
            store: Canonical local-first brain store.
            session_resolver: Callable returning stable runtime session ids.
            language: Active local runtime language.
        """
        super().__init__(name="brain-hot-memory")
        self._store = store
        self._session_resolver = session_resolver
        self._language = language
        self._last_seen_user_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Persist approved typed facts extracted from the latest user message."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            latest_user_text = latest_user_text_from_context(frame.context)
            if latest_user_text and latest_user_text != self._last_seen_user_text:
                try:
                    _record_user_turn_text(
                        store=self._store,
                        session_resolver=self._session_resolver,
                        language=self._language,
                        latest_user_text=latest_user_text,
                        source="context",
                    )
                except Exception as exc:
                    logger.warning(
                        "Hot-path memory extraction failed; continuing without frame error: %s",
                        exc,
                    )
                finally:
                    self._last_seen_user_text = latest_user_text

        await self.push_frame(frame, direction)


class BrainTextUserTurnProcessor(FrameProcessor):
    """Record text-mode user turns before the LLM executes."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
        language: Language,
        executive: BrainExecutive | None = None,
    ):
        """Initialize the text-mode user turn recorder.

        Args:
            store: Canonical local-first brain store.
            session_resolver: Callable returning stable runtime session ids.
            language: Active runtime language for hot-path memory extraction.
            executive: Optional executive reevaluated after the text turn closes.
        """
        super().__init__(name="brain-text-user-turn")
        self._store = store
        self._session_resolver = session_resolver
        self._language = language
        self._executive = executive
        self._last_seen_user_text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Record one text turn in the canonical event order."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            latest_user_text = latest_user_text_from_context(frame.context)
            if latest_user_text and latest_user_text != self._last_seen_user_text:
                session_ids = self._session_resolver()
                self._store.append_brain_event(
                    event_type=BrainEventType.USER_TURN_STARTED,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source="text",
                    payload={},
                )
                _record_user_turn_text(
                    store=self._store,
                    session_resolver=self._session_resolver,
                    language=self._language,
                    latest_user_text=latest_user_text,
                    source="text",
                )
                closed_event = self._store.append_brain_event(
                    event_type=BrainEventType.USER_TURN_ENDED,
                    agent_id=session_ids.agent_id,
                    user_id=session_ids.user_id,
                    session_id=session_ids.session_id,
                    thread_id=session_ids.thread_id,
                    source="text",
                    payload={},
                )
                if self._executive is not None:
                    self._executive.run_presence_director_reevaluation(
                        BrainReevaluationTrigger(
                            kind=BrainReevaluationConditionKind.USER_TURN_CLOSED.value,
                            summary="Reevaluate held candidates after the user turn closes.",
                            details={"turn": "user"},
                            source_event_type=closed_event.event_type,
                            source_event_id=closed_event.event_id,
                            ts=closed_event.ts,
                        )
                    )
                    await self._executive.run_commitment_wake_router(
                        boundary_kind="user_turn_end",
                        source_event=closed_event,
                    )
                self._last_seen_user_text = latest_user_text

        await self.push_frame(frame, direction)


class EmbodiedCommandProcessor(FrameProcessor):
    """Interpret explicit embodied commands before the LLM runs."""

    def __init__(
        self,
        *,
        interpreter: EmbodiedCommandInterpreter,
        action_dispatcher: EmbodiedCapabilityDispatcher,
        executive: BrainExecutive,
        session_resolver,
        store: BrainStore | None,
        presence_scope_key: str,
        language: Language,
    ):
        """Initialize the processor.

        Args:
            interpreter: Deterministic finite command interpreter.
            action_dispatcher: Canonical bounded dispatch path for single actions.
            executive: Explicit agenda/executive loop used for multi-step sequences.
            session_resolver: Callable returning stable runtime session ids.
            store: Optional local brain store for capability execution context.
            presence_scope_key: Store scope key for body-state persistence.
            language: Active runtime language for system guidance messages.
        """
        super().__init__(name="brain-embodied-command")
        self._interpreter = interpreter
        self._action_dispatcher = action_dispatcher
        self._executive = executive
        self._session_resolver = session_resolver
        self._store = store
        self._presence_scope_key = presence_scope_key
        self._language = language
        self._last_processed_text = ""
        self._message_header = "[BLINK_EMBODIED_ACTION]"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Execute one finite embodied action or inject a refusal guidance message."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            latest_user_text = latest_user_text_from_context(frame.context)
            if latest_user_text and latest_user_text != self._last_processed_text:
                filtered_messages = []
                for message in frame.context.get_messages():
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "system"
                        and isinstance(message.get("content"), str)
                        and str(message.get("content")).startswith(self._message_header)
                    ):
                        continue
                    filtered_messages.append(message)
                frame.context.set_messages(filtered_messages)
                interpretation = self._interpreter.interpret(latest_user_text)
                latest_user_turn = (
                    self._store.latest_brain_event(
                        user_id=self._session_resolver().user_id,
                        thread_id=self._session_resolver().thread_id,
                        event_types=(BrainEventType.USER_TURN_TRANSCRIBED,),
                    )
                    if self._store is not None
                    else None
                )
                if interpretation.action_id:
                    result = await self._action_dispatcher.execute_action(
                        interpretation.action_id,
                        source="interpreter",
                        reason="Deterministic embodied command interpreter matched the utterance.",
                        causal_parent_id=(
                            latest_user_turn.event_id if latest_user_turn is not None else None
                        ),
                    )
                    frame.context.add_message(
                        {
                            "role": "system",
                            "content": (
                                f"{self._message_header} 已由受控动作层执行 {interpretation.action_id}。"
                                f"结果：{result.summary}。请自然确认，不要尝试其他头部动作。"
                                if self._language.value.lower().startswith(("zh", "cmn"))
                                else f"{self._message_header} The controlled embodiment layer already executed {interpretation.action_id}. "
                                f"Result: {result.summary}. Confirm naturally and do not attempt any other head action."
                            ),
                        }
                    )
                elif interpretation.action_sequence:
                    capability_sequence = [
                        {"capability_id": capability_id_for_action(action_id)}
                        for action_id in interpretation.action_sequence
                    ]
                    goal_id = self._executive.create_goal(
                        title=latest_user_text,
                        intent="robot_head.sequence",
                        source="interpreter",
                        details={"capabilities": capability_sequence},
                        correlation_id=latest_user_turn.event_id
                        if latest_user_turn is not None
                        else None,
                        causal_parent_id=latest_user_turn.event_id
                        if latest_user_turn is not None
                        else None,
                    )
                    result = await self._executive.run_until_quiescent(
                        max_iterations=max(4, len(capability_sequence) * 3)
                    )
                    agenda = self._store.get_agenda_projection(
                        scope_key=self._session_resolver().thread_id,
                        user_id=self._session_resolver().user_id,
                    )
                    goal = agenda.goal(goal_id)
                    summary = (
                        goal.last_summary
                        if goal is not None and goal.last_summary
                        else (
                            "已通过受控执行层完成多步动作。"
                            if result.progressed
                            and self._language.value.lower().startswith(("zh", "cmn"))
                            else "The controlled execution layer handled the multi-step action."
                        )
                    )
                    frame.context.add_message(
                        {
                            "role": "system",
                            "content": (
                                f"{self._message_header} 已由执行层处理多步头部动作请求。"
                                f"结果：{summary}。请自然确认，不要追加新的头部动作。"
                                if self._language.value.lower().startswith(("zh", "cmn"))
                                else f"{self._message_header} The executive layer handled the multi-step head-action request. "
                                f"Result: {summary}. Confirm naturally and do not add new head actions."
                            ),
                        }
                    )
                elif interpretation.denied_reason:
                    reason = (
                        "用户请求了不受支持的原始或多动作头部控制。"
                        "请明确拒绝，并只允许单个有限动作。"
                        if self._language.value.lower().startswith(("zh", "cmn"))
                        else "The user requested unsupported raw or multi-action head control. "
                        "Refuse clearly and allow only one finite action at a time."
                    )
                    frame.context.add_message(
                        {
                            "role": "system",
                            "content": f"{self._message_header} {reason}",
                        }
                    )
                self._last_processed_text = latest_user_text

        await self.push_frame(frame, direction)


class BrainContextCompilerProcessor(FrameProcessor):
    """Update the leading dynamic Blink brain context system message."""

    def __init__(
        self,
        *,
        compiler: BrainContextCompiler,
        persona_modality: BrainPersonaModality | str = BrainPersonaModality.TEXT,
        packet_callback: Callable[["BrainCompiledContextPacket"], None] | None = None,
    ):
        """Initialize the processor.

        Args:
            compiler: Fixed-order context compiler for dynamic brain state.
            persona_modality: Fixed high-level modality for compact persona-expression context.
            packet_callback: Optional audit callback for the compiled packet sidecars.
        """
        super().__init__(name="brain-context-compiler")
        self._compiler = compiler
        self._persona_modality = persona_modality
        self._packet_callback = packet_callback

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Replace the prior dynamic brain context system message."""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            latest_user_text = latest_user_text_from_context(frame.context)
            messages = list(frame.context.get_messages())
            filtered_messages = []
            for message in messages:
                if (
                    isinstance(message, dict)
                    and message.get("role") == "system"
                    and isinstance(message.get("content"), str)
                    and str(message.get("content")).startswith(BRAIN_CONTEXT_HEADER)
                ):
                    continue
                filtered_messages.append(message)

            packet = self._compiler.compile_packet(
                latest_user_text=latest_user_text,
                persona_modality=self._persona_modality,
            )
            if self._packet_callback is not None:
                self._packet_callback(packet)
            filtered_messages.insert(
                0,
                {
                    "role": "system",
                    "content": packet.prompt,
                },
            )
            frame.context.set_messages(filtered_messages)

        await self.push_frame(frame, direction)


class BrainExecutiveStartupProcessor(FrameProcessor):
    """Run one bounded executive startup reevaluation on pipeline start."""

    def __init__(self, *, executive: BrainExecutive):
        """Initialize the processor."""
        super().__init__(name="brain-executive-startup")
        self._executive = executive
        self._ran_startup_pass = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Run the startup pass exactly once on the first StartFrame."""
        await super().process_frame(frame, direction)
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, StartFrame)
            and not self._ran_startup_pass
        ):
            await self._executive.run_startup_pass()
            self._ran_startup_pass = True
        await self.push_frame(frame, direction)


class TurnRecorderProcessor(FrameProcessor):
    """Persist turn episodes after assistant context has been updated."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
        context: LLMContext,
        consolidator: BrainMemoryConsolidator,
        executive: BrainExecutive | None = None,
        memory_use_trace_resolver: Callable[[], "BrainMemoryUseTrace | None"] | None = None,
        memory_use_trace_committed_callback: Callable[["BrainMemoryUseTrace"], None] | None = None,
    ):
        """Initialize the turn recorder.

        Args:
            store: Canonical local-first brain store.
            session_resolver: Callable returning stable runtime session ids.
            context: Shared final LLM context for the active conversation.
            consolidator: Background consolidator for thread summaries.
            executive: Optional durable executive reevaluated at turn end.
            memory_use_trace_resolver: Optional resolver for the pending context trace.
            memory_use_trace_committed_callback: Optional callback for persisted traces.
        """
        super().__init__(name="brain-turn-recorder")
        self._turn_persister = _BrainTurnPersistenceHelper(
            store=store,
            session_resolver=session_resolver,
            context=context,
            consolidator=consolidator,
            executive=executive,
            memory_use_trace_resolver=memory_use_trace_resolver,
            memory_use_trace_committed_callback=memory_use_trace_committed_callback,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Persist completed conversation turns after Blink finishes speaking."""
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, BotStoppedSpeakingFrame):
            try:
                await self._turn_persister.persist_completed_turn(source="turn-recorder")
            except Exception as exc:
                logger.warning(
                    "Brain turn persistence failed; continuing realtime pipeline: %s",
                    type(exc).__name__,
                )

        await self.push_frame(frame, direction)


class BrainTextAssistantTurnStartRecorder(FrameProcessor):
    """Record text-mode assistant start boundaries before aggregation."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
    ):
        """Initialize the text-mode assistant start recorder.

        Args:
            store: Canonical local-first brain store.
            session_resolver: Callable returning stable runtime session ids.
        """
        super().__init__(name="brain-text-turn-start-recorder")
        self._store = store
        self._session_resolver = session_resolver

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Append assistant turn start events for text-mode replies."""
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMFullResponseStartFrame):
            session_ids = self._session_resolver()
            self._store.append_brain_event(
                event_type=BrainEventType.ASSISTANT_TURN_STARTED,
                agent_id=session_ids.agent_id,
                user_id=session_ids.user_id,
                session_id=session_ids.session_id,
                thread_id=session_ids.thread_id,
                source="text",
                payload={},
            )

        await self.push_frame(frame, direction)


class BrainTextAssistantTurnFinalizer(FrameProcessor):
    """Persist text-mode assistant turns after context aggregation completes."""

    def __init__(
        self,
        *,
        store: BrainStore,
        session_resolver,
        context: LLMContext,
        consolidator: BrainMemoryConsolidator,
        executive: BrainExecutive | None = None,
        memory_use_trace_resolver: Callable[[], "BrainMemoryUseTrace | None"] | None = None,
        memory_use_trace_committed_callback: Callable[["BrainMemoryUseTrace"], None] | None = None,
    ):
        """Initialize the text-mode assistant turn finalizer.

        Args:
            store: Canonical local-first brain store.
            session_resolver: Callable returning stable runtime session ids.
            context: Shared final LLM context for the active text conversation.
            consolidator: Background consolidator for thread summaries.
            executive: Optional durable executive reevaluated at turn end.
            memory_use_trace_resolver: Optional resolver for the pending context trace.
            memory_use_trace_committed_callback: Optional callback for persisted traces.
        """
        super().__init__(name="brain-text-turn-recorder")
        self._turn_persister = _BrainTurnPersistenceHelper(
            store=store,
            session_resolver=session_resolver,
            context=context,
            consolidator=consolidator,
            executive=executive,
            memory_use_trace_resolver=memory_use_trace_resolver,
            memory_use_trace_committed_callback=memory_use_trace_committed_callback,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Persist text turns once the assistant context has finished updating."""
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(
            frame, LLMContextAssistantTimestampFrame
        ):
            await self._turn_persister.persist_completed_turn(source="text")

        await self.push_frame(frame, direction)

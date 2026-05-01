"""Browser/WebRTC interruption policy and public-safe decision tracking."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from blink.frames.frames import (
    AggregatedTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from blink.interaction.floor import (
    ConversationFloorTextKind,
    classify_floor_text,
    is_short_backchannel,
)
from blink.interaction.performance_events import BrowserInteractionMode
from blink.processors.frame_processor import FrameDirection, FrameProcessor
from blink.turns.types import ProcessFrameResult
from blink.turns.user_mute import AlwaysUserMuteStrategy
from blink.turns.user_start import BaseUserTurnStartStrategy

BrowserPerformanceEmit = Callable[..., object]

_FALSE_INTERRUPTION_REASONS = {
    "short_backchannel",
    "brief_speech_or_cough",
    "acoustic_noise",
    "low_confidence_transcript",
    "protected_playback",
    "assistant_not_speaking",
}
_COUGH_OR_NOISE_TOKENS = {
    "咳",
    "咳嗽",
    "咳咳",
    "咳了一下",
    "cough",
    "coughing",
    "noise",
    "background_noise",
    "backgroundnoise",
}
_LOW_CONFIDENCE_THRESHOLD = 0.35

def _safe_reason_codes(values: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()[:96]
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= 16:
            break
    return result


def _safe_token(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = str(getattr(value, "value", value) or "").strip().replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in {"_", "-", ".", ":"} else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < _LOW_CONFIDENCE_THRESHOLD:
        return "low"
    if value < 0.7:
        return "medium"
    return "high"


def _is_cough_or_noise(value: str) -> bool:
    kind = classify_floor_text(value)
    if kind == ConversationFloorTextKind.EMPTY:
        return True
    normalized = "".join(ch.lower() for ch in str(value or "") if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
    return normalized in _COUGH_OR_NOISE_TOKENS


def _frame_confidence(frame: Frame) -> float | None:
    direct = _safe_float(getattr(frame, "confidence", None))
    if direct is not None:
        return direct
    result = getattr(frame, "result", None)
    if isinstance(result, dict):
        for key in ("confidence", "transcript_confidence", "language_probability"):
            value = _safe_float(result.get(key))
            if value is not None:
                return value
    return None


@dataclass(frozen=True)
class InterruptionDecisionContext:
    """Public-safe inputs for deterministic interruption decisions."""

    floor_state: str = "unknown"
    assistant_speaking: bool = False
    speech_age_ms: int = 0
    vad_duration_ms: int = 0
    transcription_chars: int = 0
    transcript_confidence: float | None = None
    text_kind: str = "empty"
    echo_risk: str = "unknown"
    barge_in_state: str = "protected"
    protected_playback: bool = True
    reason_codes: tuple[str, ...] = ()

    def as_metadata(self) -> dict[str, Any]:
        """Return public-safe metadata for interruption events."""
        metadata: dict[str, Any] = {
            "floor_state": _safe_token(self.floor_state),
            "assistant_speaking": self.assistant_speaking is True,
            "speech_age_ms": max(0, int(self.speech_age_ms)),
            "vad_duration_ms": max(0, int(self.vad_duration_ms)),
            "text_kind": _safe_token(self.text_kind, default="empty"),
            "echo_risk_state": _safe_token(self.echo_risk),
            "barge_in_state": _safe_token(self.barge_in_state),
            "protected_playback_enabled": self.protected_playback is True,
            "stt_confidence_bucket": _confidence_bucket(self.transcript_confidence),
        }
        if self.transcription_chars:
            metadata["transcription_chars"] = max(0, int(self.transcription_chars))
        if self.transcript_confidence is not None:
            metadata["stt_confidence"] = round(
                max(0.0, min(1.0, self.transcript_confidence)),
                3,
            )
        return metadata

    def event_reason_codes(self, *values: str) -> tuple[str, ...]:
        """Return public-safe reason codes for an interruption event."""
        return tuple(
            _safe_reason_codes(
                (
                    f"floor_state:{self.floor_state}",
                    f"text_kind:{self.text_kind}",
                    f"echo_risk:{self.echo_risk}",
                    f"barge_in:{self.barge_in_state}",
                    *self.reason_codes,
                    *values,
                )
            )
        )


@dataclass
class BrowserInterruptionStateTracker:
    """Public-safe browser interruption state and event recorder."""

    protected_playback: bool = True
    performance_emit: BrowserPerformanceEmit | None = None
    last_decision: str = "none"
    last_reason: str = "none"
    last_event_at: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    policy_state: str | None = None
    echo_risk: str = "unknown"

    def __post_init__(self):
        self._counts: Counter[str] = Counter()
        self._false_interruption_counts: Counter[str] = Counter()
        self._lock = Lock()

    @property
    def barge_in_state(self) -> str:
        """Return the public barge-in state."""
        if self.policy_state in {"protected", "armed", "adaptive"}:
            return str(self.policy_state)
        return "protected" if self.protected_playback else "armed"

    def set_protected_playback(self, value: bool) -> None:
        """Update the active protected playback mode."""
        with self._lock:
            self.protected_playback = bool(value)
            if self.policy_state == "armed" and self.protected_playback:
                self.policy_state = None

    def set_barge_in_policy_state(self, value: str | None) -> None:
        """Update the public interruption policy state."""
        state = str(value or "").strip().lower()
        with self._lock:
            self.policy_state = state if state in {"protected", "armed", "adaptive"} else None

    def set_audio_health_policy(self, *, barge_in_state: str | None, echo_risk: str | None) -> None:
        """Update interruption policy hints from WebRTC audio health."""
        policy_state = str(barge_in_state or "").strip().lower()
        risk_state = str(echo_risk or "").strip().lower()
        with self._lock:
            self.policy_state = (
                policy_state if policy_state in {"protected", "armed", "adaptive"} else None
            )
            self.echo_risk = (
                risk_state if risk_state in {"unknown", "low", "medium", "high"} else "unknown"
            )

    def snapshot(self) -> dict[str, Any]:
        """Return the public interruption snapshot."""
        with self._lock:
            state = self.barge_in_state
            effective_protected = state == "protected"
            reason_codes = _safe_reason_codes(
                (
                    "interruption_state:v1",
                    f"interruption:{state}",
                    f"interruption_last_decision:{self.last_decision}",
                    f"interruption_last_reason:{self.last_reason}",
                    *tuple(self.reason_codes),
                )
            )
            return {
                "schema_version": 1,
                "protected_playback": effective_protected,
                "configured_protected_playback": self.protected_playback,
                "barge_in_state": state,
                "armed": state in {"armed", "adaptive"},
                "adaptive": state == "adaptive",
                "headphones_recommended": state == "armed",
                "echo_risk": self.echo_risk,
                "last_decision": self.last_decision,
                "last_reason": self.last_reason,
                "last_event_at": self.last_event_at,
                "counts": dict(sorted(self._counts.items())),
                "false_interruption_counts": dict(
                    sorted(self._false_interruption_counts.items())
                ),
                "reason_codes": reason_codes,
            }

    def record_candidate(
        self,
        *,
        speech_age_ms: int = 0,
        speech_threshold_ms: int = 450,
        context: InterruptionDecisionContext | None = None,
    ):
        """Record a candidate interruption."""
        metadata = {
            "speech_age_ms": speech_age_ms,
            "speech_threshold_ms": speech_threshold_ms,
        }
        reason_codes: tuple[str, ...] = ()
        if context is not None:
            metadata.update(context.as_metadata())
            reason_codes = context.event_reason_codes("interruption:candidate")
        self._record(
            "candidate",
            event_type="interruption.candidate",
            mode=BrowserInteractionMode.SPEAKING,
            reason="candidate_user_speech",
            metadata=metadata,
            extra_reason_codes=reason_codes,
        )

    def record_accepted(
        self,
        *,
        reason: str,
        speech_age_ms: int = 0,
        speech_threshold_ms: int = 450,
        transcription_chars: int = 0,
        context: InterruptionDecisionContext | None = None,
    ):
        """Record an accepted interruption."""
        metadata = {
            "speech_age_ms": speech_age_ms,
            "speech_threshold_ms": speech_threshold_ms,
        }
        if transcription_chars:
            metadata["transcription_chars"] = transcription_chars
        reason_codes: tuple[str, ...] = ()
        if context is not None:
            metadata.update(context.as_metadata())
            reason_codes = context.event_reason_codes(f"interruption_reason:{reason}")
        self._record(
            "accepted",
            event_type="interruption.accepted",
            mode=BrowserInteractionMode.INTERRUPTED,
            reason=reason,
            metadata=metadata,
            extra_reason_codes=reason_codes,
        )

    def record_rejected(
        self,
        *,
        reason: str,
        speech_age_ms: int = 0,
        speech_threshold_ms: int = 450,
        transcription_chars: int = 0,
        context: InterruptionDecisionContext | None = None,
    ):
        """Record a rejected interruption candidate."""
        metadata = {
            "speech_age_ms": speech_age_ms,
            "speech_threshold_ms": speech_threshold_ms,
        }
        if transcription_chars:
            metadata["transcription_chars"] = transcription_chars
        reason_codes: tuple[str, ...] = ()
        if context is not None:
            metadata.update(context.as_metadata())
            reason_codes = context.event_reason_codes(f"interruption_reason:{reason}")
        self._record(
            "rejected",
            event_type="interruption.rejected",
            mode=BrowserInteractionMode.SPEAKING,
            reason=reason,
            metadata=metadata,
            extra_reason_codes=reason_codes,
        )

    def record_suppressed(
        self,
        *,
        reason: str,
        context: InterruptionDecisionContext | None = None,
    ):
        """Record an interruption suppressed by protected playback."""
        metadata = {"protected_playback_enabled": True}
        reason_codes: tuple[str, ...] = ()
        if context is not None:
            metadata.update(context.as_metadata())
            reason_codes = context.event_reason_codes(f"interruption_reason:{reason}")
        self._record(
            "suppressed",
            event_type="interruption.suppressed",
            mode=BrowserInteractionMode.SPEAKING,
            reason=reason,
            metadata=metadata,
            extra_reason_codes=reason_codes,
        )

    def record_output_dropped(self, *, frame_type: str, output_dropped_count: int = 1):
        """Record stale output dropped after an accepted interruption."""
        self._record(
            "output_dropped",
            event_type="interruption.output_dropped",
            mode=BrowserInteractionMode.SPEAKING,
            reason="stale_output",
            metadata={
                "frame_type": frame_type,
                "output_dropped_count": max(1, int(output_dropped_count)),
            },
        )

    def record_output_flushed(self, *, frame_type: str, output_flushed_count: int = 1):
        """Record stale output flushed after an accepted interruption.

        Emits the older ``interruption.output_dropped`` event first for
        compatibility, then the v2 public ``interruption.output_flushed`` event.
        """
        count = max(1, int(output_flushed_count))
        metadata = {
            "frame_type": frame_type,
            "output_flushed_count": count,
        }
        compat_metadata = {
            "frame_type": frame_type,
            "output_dropped_count": count,
        }
        reason_codes = _safe_reason_codes(
            (
                "interruption:output_flushed",
                "interruption_reason:stale_output",
                f"interruption_mode:{self.barge_in_state}",
            )
        )
        self._emit_performance_event(
            event_type="interruption.output_dropped",
            mode=BrowserInteractionMode.SPEAKING,
            metadata=compat_metadata,
            reason_codes=reason_codes,
        )
        self._record(
            "output_flushed",
            event_type="interruption.output_flushed",
            mode=BrowserInteractionMode.SPEAKING,
            reason="stale_output",
            metadata=metadata,
            extra_reason_codes=tuple(reason_codes),
        )

    def record_listening_resumed(self, *, reason: str = "bot_speech_ended"):
        """Record that interruption handling returned to listening."""
        self._record(
            "listening_resumed",
            event_type="interruption.listening_resumed",
            mode=BrowserInteractionMode.LISTENING,
            reason=reason,
            metadata={},
        )

    def _record(
        self,
        decision: str,
        *,
        event_type: str,
        mode: BrowserInteractionMode,
        reason: str,
        metadata: dict[str, Any],
        extra_reason_codes: tuple[str, ...] = (),
    ) -> None:
        reason_codes = _safe_reason_codes(
            (
                f"interruption:{decision}",
                f"interruption_reason:{reason}",
                f"interruption_mode:{self.barge_in_state}",
                *extra_reason_codes,
            )
        )
        with self._lock:
            self.last_decision = decision
            self.last_reason = reason
            self._counts[decision] += 1
            if decision in {"rejected", "suppressed"} or reason in _FALSE_INTERRUPTION_REASONS:
                self._false_interruption_counts[reason] += 1
            self.reason_codes = reason_codes
        event = self._emit_performance_event(
            event_type=event_type,
            mode=mode,
            metadata=metadata,
            reason_codes=reason_codes,
        )
        if event is not None:
            timestamp = getattr(event, "timestamp", None)
            if timestamp:
                with self._lock:
                    self.last_event_at = str(timestamp)

    def _emit_performance_event(
        self,
        *,
        event_type: str,
        mode: BrowserInteractionMode,
        metadata: dict[str, Any],
        reason_codes: list[str],
    ) -> object | None:
        if self.performance_emit is None:
            return None
        return self.performance_emit(
            event_type=event_type,
            source="interruption",
            mode=mode,
            metadata=metadata,
            reason_codes=tuple(reason_codes),
        )


class BrowserProtectedPlaybackMuteStrategy(AlwaysUserMuteStrategy):
    """Protected playback mute strategy with public-safe interruption accounting."""

    def __init__(self, *, interruption_state: BrowserInterruptionStateTracker):
        """Initialize the strategy."""
        super().__init__()
        self._interruption_state = interruption_state
        self._suppressed_during_current_speech = False

    async def reset(self):
        """Reset protected mute state."""
        await super().reset()
        self._suppressed_during_current_speech = False

    async def process_frame(self, frame: Frame) -> bool:
        """Mute user input while bot speech is protected and record suppressions."""
        was_bot_speaking = self._bot_speaking
        muted = await super().process_frame(frame)
        if self._interruption_state.barge_in_state != "protected":
            return False
        if was_bot_speaking and isinstance(
            frame,
            (
                VADUserStartedSpeakingFrame,
                InterimTranscriptionFrame,
                TranscriptionFrame,
            ),
        ):
            context = InterruptionDecisionContext(
                assistant_speaking=True,
                text_kind=(
                    classify_floor_text(str(getattr(frame, "text", "") or "")).value
                    if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame))
                    else "empty"
                ),
                transcription_chars=len(str(getattr(frame, "text", "") or "").strip()),
                transcript_confidence=_frame_confidence(frame),
                barge_in_state=self._interruption_state.barge_in_state,
                protected_playback=True,
            )
            self._interruption_state.record_suppressed(
                reason="protected_playback",
                context=context,
            )
            self._suppressed_during_current_speech = True
        if isinstance(frame, BotStoppedSpeakingFrame) and self._suppressed_during_current_speech:
            self._interruption_state.record_listening_resumed(reason="protected_playback_released")
            self._suppressed_during_current_speech = False
        return muted


class BrowserBargeInTurnStartStrategy(BaseUserTurnStartStrategy):
    """Browser-only turn start strategy for explicit measured barge-in."""

    def __init__(
        self,
        *,
        interruption_state: BrowserInterruptionStateTracker,
        sustain_threshold_secs: float = 0.45,
        **kwargs,
    ):
        """Initialize the strategy."""
        super().__init__(**kwargs)
        self._interruption_state = interruption_state
        self._sustain_threshold_secs = max(0.0, float(sustain_threshold_secs))
        self._bot_speaking = False
        self._accepted = False
        self._pending_started_at: float | None = None
        self._pending_accept_task: asyncio.Task | None = None
        self._activity_during_bot_speech = False

    async def cleanup(self):
        """Cancel pending sustained-speech checks."""
        await self._cancel_pending_accept()
        await super().cleanup()

    async def reset(self):
        """Reset candidate state for the next user turn."""
        await self._cancel_pending_accept()
        self._accepted = False
        self._pending_started_at = None

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process browser speech events into conservative interruption decisions."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._activity_during_bot_speech = False
            return ProcessFrameResult.CONTINUE
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            if self._pending_started_at is not None and not self._accepted:
                self._activity_during_bot_speech = True
                self._interruption_state.record_rejected(
                    reason="bot_speech_ended",
                    speech_age_ms=self._pending_speech_age_ms(),
                    speech_threshold_ms=self._threshold_ms(),
                    context=self._decision_context(
                        speech_age_ms=self._pending_speech_age_ms(),
                        reason_codes=("interruption:bot_speech_ended",),
                    ),
                )
                await self._cancel_pending_accept()
            if self._activity_during_bot_speech:
                self._interruption_state.record_listening_resumed()
                self._activity_during_bot_speech = False
            return ProcessFrameResult.CONTINUE
        if isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._bot_speaking:
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            if self._pending_started_at is None and not self._accepted:
                self._pending_started_at = time.monotonic()
                self._activity_during_bot_speech = True
                context = self._decision_context(reason_codes=("interruption:vad_candidate",))
                self._interruption_state.record_candidate(
                    speech_age_ms=0,
                    speech_threshold_ms=self._threshold_ms(),
                    context=context,
                )
                self._pending_accept_task = self.task_manager.create_task(
                    self._accept_pending_after_delay(),
                    f"{self}::_accept_pending_after_delay",
                )
            return ProcessFrameResult.STOP
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._pending_started_at is not None and not self._accepted:
                self._activity_during_bot_speech = True
                speech_age_ms = self._pending_speech_age_ms()
                reason = "brief_speech_or_cough"
                self._interruption_state.record_rejected(
                    reason=reason,
                    speech_age_ms=speech_age_ms,
                    speech_threshold_ms=self._threshold_ms(),
                    context=self._decision_context(
                        speech_age_ms=speech_age_ms,
                        vad_duration_ms=speech_age_ms,
                        reason_codes=(f"interruption:{reason}",),
                    ),
                )
                await self._cancel_pending_accept()
            return ProcessFrameResult.STOP if self._bot_speaking else ProcessFrameResult.CONTINUE
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)):
            user_text = str(getattr(frame, "text", "") or "")
            if not user_text.strip():
                return ProcessFrameResult.CONTINUE
            if not self._bot_speaking:
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            text_kind = classify_floor_text(user_text)
            confidence = _frame_confidence(frame)
            context = self._decision_context(
                text_kind=text_kind.value,
                transcription_chars=len(user_text.strip()),
                transcript_confidence=confidence,
            )
            if _is_cough_or_noise(user_text):
                self._activity_during_bot_speech = True
                self._interruption_state.record_rejected(
                    reason="acoustic_noise",
                    speech_age_ms=self._pending_speech_age_ms(),
                    speech_threshold_ms=self._threshold_ms(),
                    transcription_chars=len(user_text.strip()),
                    context=context,
                )
                await self._cancel_pending_accept()
                await self.trigger_reset_aggregation()
                return ProcessFrameResult.STOP
            if confidence is not None and confidence < _LOW_CONFIDENCE_THRESHOLD:
                self._activity_during_bot_speech = True
                self._interruption_state.record_rejected(
                    reason="low_confidence_transcript",
                    speech_age_ms=self._pending_speech_age_ms(),
                    speech_threshold_ms=self._threshold_ms(),
                    transcription_chars=len(user_text.strip()),
                    context=context,
                )
                await self._cancel_pending_accept()
                await self.trigger_reset_aggregation()
                return ProcessFrameResult.STOP
            if is_short_backchannel(user_text):
                self._activity_during_bot_speech = True
                self._interruption_state.record_rejected(
                    reason="short_backchannel",
                    speech_age_ms=self._pending_speech_age_ms(),
                    speech_threshold_ms=self._threshold_ms(),
                    transcription_chars=len(user_text.strip()),
                    context=context,
                )
                await self._cancel_pending_accept()
                await self.trigger_reset_aggregation()
                return ProcessFrameResult.STOP
            await self._accept(
                reason=(
                    "explicit_interruption"
                    if text_kind == ConversationFloorTextKind.EXPLICIT_INTERRUPTION
                    else "meaningful_transcription"
                ),
                transcription_chars=len(user_text.strip()),
                context=context,
            )
            return ProcessFrameResult.STOP
        return ProcessFrameResult.CONTINUE

    async def _accept_pending_after_delay(self) -> None:
        await asyncio.sleep(self._sustain_threshold_secs)
        if self._accepted or self._pending_started_at is None:
            return
        if not self._bot_speaking:
            self._activity_during_bot_speech = True
            self._interruption_state.record_rejected(
                reason="bot_speech_ended",
                speech_age_ms=self._pending_speech_age_ms(),
                speech_threshold_ms=self._threshold_ms(),
                context=self._decision_context(
                    speech_age_ms=self._pending_speech_age_ms(),
                    reason_codes=("interruption:bot_speech_ended",),
                ),
            )
            self._pending_started_at = None
            self._pending_accept_task = None
            return
        await self._accept(
            reason="sustained_speech",
            context=self._decision_context(
                speech_age_ms=self._pending_speech_age_ms(),
                vad_duration_ms=self._pending_speech_age_ms(),
                reason_codes=("interruption:sustained_speech",),
            ),
        )

    async def _accept(
        self,
        *,
        reason: str,
        transcription_chars: int = 0,
        context: InterruptionDecisionContext | None = None,
    ) -> None:
        if self._accepted:
            return
        self._accepted = True
        self._activity_during_bot_speech = True
        speech_age_ms = self._pending_speech_age_ms()
        pending_task = self._pending_accept_task
        self._pending_started_at = None
        self._pending_accept_task = None
        if (
            pending_task is not None
            and not pending_task.done()
            and pending_task is not asyncio.current_task()
        ):
            await asyncio.sleep(0)
            await self.task_manager.cancel_task(pending_task)
        self._interruption_state.record_accepted(
            reason=reason,
            speech_age_ms=speech_age_ms,
            speech_threshold_ms=self._threshold_ms(),
            transcription_chars=transcription_chars,
            context=context
            or self._decision_context(
                speech_age_ms=speech_age_ms,
                transcription_chars=transcription_chars,
                reason_codes=(f"interruption:{reason}",),
            ),
        )
        await self.trigger_user_turn_started()

    def _decision_context(
        self,
        *,
        speech_age_ms: int | None = None,
        vad_duration_ms: int = 0,
        transcription_chars: int = 0,
        transcript_confidence: float | None = None,
        text_kind: str = "empty",
        reason_codes: tuple[str, ...] = (),
    ) -> InterruptionDecisionContext:
        state_snapshot = self._interruption_state.snapshot()
        return InterruptionDecisionContext(
            assistant_speaking=self._bot_speaking,
            speech_age_ms=self._pending_speech_age_ms()
            if speech_age_ms is None
            else max(0, int(speech_age_ms)),
            vad_duration_ms=max(0, int(vad_duration_ms)),
            transcription_chars=max(0, int(transcription_chars)),
            transcript_confidence=transcript_confidence,
            text_kind=text_kind,
            echo_risk=str(state_snapshot.get("echo_risk", "unknown")),
            barge_in_state=str(state_snapshot.get("barge_in_state", "protected")),
            protected_playback=state_snapshot.get("protected_playback") is True,
            reason_codes=reason_codes,
        )

    async def _cancel_pending_accept(self) -> None:
        task = self._pending_accept_task
        self._pending_accept_task = None
        self._pending_started_at = None
        if task is None or task.done():
            return
        await asyncio.sleep(0)
        await self.task_manager.cancel_task(task)

    def _pending_speech_age_ms(self) -> int:
        if self._pending_started_at is None:
            return 0
        return max(0, int((time.monotonic() - self._pending_started_at) * 1000))

    def _threshold_ms(self) -> int:
        return max(0, int(self._sustain_threshold_secs * 1000))


class BrowserInterruptedOutputGuardProcessor(FrameProcessor):
    """Drop stale assistant text after an accepted browser interruption."""

    def __init__(
        self,
        *,
        interruption_state: BrowserInterruptionStateTracker,
        metrics_recorder: Any | None = None,
        performance_emit: BrowserPerformanceEmit | None = None,
    ):
        """Initialize the output guard."""
        super().__init__(name="browser-interrupted-output-guard")
        self._interruption_state = interruption_state
        self._metrics_recorder = metrics_recorder
        self._performance_emit = performance_emit
        self._dropping_interrupted_output = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Drop stale text/TTS frames until the next assistant response starts."""
        await super().process_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, InterruptionFrame):
            self._dropping_interrupted_output = True
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, LLMFullResponseStartFrame):
            self._dropping_interrupted_output = False
            await self.push_frame(frame, direction)
            return
        if self._dropping_interrupted_output and self._should_drop_frame(frame):
            self._interruption_state.record_output_flushed(frame_type=frame.__class__.__name__)
            if getattr(frame, "blink_speech_chunk", None) is not None:
                recorder = getattr(self._metrics_recorder, "record_stale_chunk_drop", None)
                if callable(recorder):
                    recorder()
                if self._performance_emit is not None:
                    chunk = getattr(frame, "blink_speech_chunk", None)
                    metadata = (
                        chunk.public_metadata()
                        if hasattr(chunk, "public_metadata")
                        else {
                            "chunk_id": getattr(chunk, "chunk_id", "unknown"),
                            "language": getattr(chunk, "language", "unknown"),
                            "tts_backend": getattr(chunk, "tts_backend", "unknown"),
                            "generation_id": getattr(chunk, "generation_id", "unknown"),
                            "turn_id": getattr(chunk, "turn_id", "unknown"),
                            "chunk_index": getattr(chunk, "chunk_index", 0),
                            "display_chars": getattr(chunk, "display_chars", 0),
                            "speak_chars": getattr(chunk, "speak_chars", 0),
                        }
                    )
                    self._performance_emit(
                        event_type="speech.chunk_stale_dropped",
                        source="speech_director",
                        mode=BrowserInteractionMode.INTERRUPTED,
                        metadata=metadata,
                        reason_codes=("speech:stale_chunk_dropped",),
                    )
            return
        await self.push_frame(frame, direction)

    @staticmethod
    def _should_drop_frame(frame: Frame) -> bool:
        if isinstance(frame, TTSSpeakFrame):
            return True
        if isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            return False
        return isinstance(frame, (TextFrame, AggregatedTextFrame))


__all__ = [
    "BrowserBargeInTurnStartStrategy",
    "BrowserInterruptedOutputGuardProcessor",
    "BrowserInterruptionStateTracker",
    "BrowserProtectedPlaybackMuteStrategy",
    "InterruptionDecisionContext",
]

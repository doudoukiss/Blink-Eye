"""Public-safe browser performance events."""

from __future__ import annotations

import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from typing import Any, Callable, Optional

from blink.interaction.actor_control_frame_v3 import ActorControlFrameV3, ActorControlScheduler
from blink.interaction.actor_events import (
    ActorEventContext,
    ActorEventV2,
    ActorTraceWriter,
    actor_event_from_performance_event,
)
from blink.interaction.performance_episode_v3 import PerformanceEpisodeV3Writer

logger = logging.getLogger(__name__)


class BrowserInteractionMode(str, Enum):
    """High-level public browser interaction modes."""

    WAITING = "waiting"
    CONNECTED = "connected"
    LISTENING = "listening"
    HEARD = "heard"
    THINKING = "thinking"
    LOOKING = "looking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


_TOKEN_RE = re.compile(r"[^a-zA-Z0-9_.:-]+")
_SENSITIVE_KEY_FRAGMENTS = {
    "authorization",
    "audio",
    "candidate",
    "credential",
    "example",
    "image",
    "memory_id",
    "password",
    "prompt",
    "raw",
    "sdp",
    "secret",
    "source_ref",
    "text",
    "token",
    "transcript",
    "url",
}
_SENSITIVE_EXACT_KEYS = {
    "audio",
    "bytes",
    "content",
    "db_path",
    "ice",
    "ice_candidate",
    "messages",
    "records",
}
_SAFE_TEXT_KEY_SUFFIXES = (
    "_chars",
    "_count",
    "_counts",
    "_enabled",
    "_hash",
    "_kind",
    "_ids",
    "_ms",
    "_state",
)
_SAFE_TEXT_KEY_EXACT = {"last_text_kind", "stale_generation_token", "text_kind"}


def _safe_token(value: object, *, default: str = "unknown", limit: int = 96) -> str:
    text = _TOKEN_RE.sub("_", str(value or "").strip())
    text = "_".join(part for part in text.split("_") if part)
    return text[:limit] or default


def _safe_optional_token(value: object, *, limit: int = 96) -> str | None:
    if value in (None, ""):
        return None
    return _safe_token(value, limit=limit)


def _safe_reason_codes(values: object, *, limit: int = 16) -> list[str]:
    raw_values = values if isinstance(values, (list, tuple, set)) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        code = _safe_token(raw_value, default="", limit=96)
        if not code or code in seen:
            continue
        seen.add(code)
        result.append(code)
        if len(result) >= limit:
            break
    return result


def _metadata_key_is_safe(key: object) -> bool:
    text = _safe_token(key, default="", limit=80).lower()
    if not text or text in _SENSITIVE_EXACT_KEYS:
        return False
    if text in _SAFE_TEXT_KEY_EXACT:
        return True
    if text.endswith(_SAFE_TEXT_KEY_SUFFIXES):
        return True
    return not any(fragment in text for fragment in _SENSITIVE_KEY_FRAGMENTS)


def _safe_text(value: object, *, limit: int = 160) -> str:
    return " ".join(str(value or "").split())[:limit]


def _sanitize_metadata_value(value: object, *, depth: int = 0) -> object:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _safe_text(value)
    if depth >= 3:
        return _safe_text(type(value).__name__, limit=80)
    if isinstance(value, (list, tuple, set)):
        return [
            sanitized
            for item in list(value)[:12]
            if (sanitized := _sanitize_metadata_value(item, depth=depth + 1)) is not None
        ]
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for raw_key, raw_item in list(value.items())[:24]:
            if not _metadata_key_is_safe(raw_key):
                continue
            key = _safe_token(raw_key, default="", limit=80)
            if not key:
                continue
            sanitized = _sanitize_metadata_value(raw_item, depth=depth + 1)
            if sanitized is not None:
                result[key] = sanitized
        return result
    return _safe_text(type(value).__name__, limit=80)


def sanitize_metadata(metadata: object) -> dict[str, object]:
    """Return bounded public-safe event metadata."""
    if not isinstance(metadata, dict):
        return {}
    sanitized = _sanitize_metadata_value(metadata)
    return sanitized if isinstance(sanitized, dict) else {}


@dataclass(frozen=True)
class BrowserPerformanceEvent:
    """One public-safe browser performance event."""

    event_id: int
    event_type: str
    source: str
    mode: BrowserInteractionMode
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    session_id: Optional[str] = None
    client_id: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return the event as a public API payload."""
        return {
            "schema_version": 1,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "mode": self.mode.value,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "client_id": self.client_id,
            "metadata": dict(self.metadata),
            "reason_codes": list(self.reason_codes),
        }


class BrowserPerformanceEventBus:
    """Bounded in-memory event bus for browser runtime performance state."""

    def __init__(
        self,
        *,
        max_events: int = 500,
        actor_context_provider: Callable[[], ActorEventContext | dict[str, object]] | None = None,
        actor_trace_writer: ActorTraceWriter | None = None,
        performance_episode_writer: PerformanceEpisodeV3Writer | None = None,
        discourse_episode_collector: Any | None = None,
        actor_control_scheduler: ActorControlScheduler | None = None,
        max_actor_events: int | None = None,
        max_actor_control_frames: int | None = None,
    ):
        """Initialize the event bus.

        Args:
            max_events: Maximum number of recent events to retain.
            actor_context_provider: Optional v2 actor event runtime context provider.
            actor_trace_writer: Optional bounded JSONL trace writer for actor events.
            performance_episode_writer: Optional bounded JSONL performance episode writer.
            discourse_episode_collector: Optional internal discourse episode collector.
            actor_control_scheduler: Optional internal control-frame scheduler.
            max_actor_events: Optional maximum number of recent v2 events to retain.
            max_actor_control_frames: Optional maximum number of retained control frames.
        """
        self._events: deque[BrowserPerformanceEvent] = deque(maxlen=max(1, int(max_events)))
        self._actor_events: deque[ActorEventV2] = deque(
            maxlen=max(1, int(max_actor_events or max_events))
        )
        self._actor_control_frames: deque[ActorControlFrameV3] = deque(
            maxlen=max(1, int(max_actor_control_frames or max_events))
        )
        self._actor_context_provider = actor_context_provider
        self._actor_trace_writer = actor_trace_writer
        self._performance_episode_writer = performance_episode_writer
        self._discourse_episode_collector = discourse_episode_collector
        self._actor_control_scheduler = actor_control_scheduler
        self._next_event_id = 1
        self._lock = Lock()

    @property
    def latest_event_id(self) -> int:
        """Return the latest emitted event id, or 0 when empty."""
        with self._lock:
            return self._events[-1].event_id if self._events else 0

    @property
    def latest_event(self) -> BrowserPerformanceEvent | None:
        """Return the latest event, if any."""
        with self._lock:
            return self._events[-1] if self._events else None

    @property
    def actor_latest_event_id(self) -> int:
        """Return the latest emitted actor event id, or 0 when empty."""
        with self._lock:
            return self._actor_events[-1].event_id if self._actor_events else 0

    @property
    def actor_latest_event(self) -> ActorEventV2 | None:
        """Return the latest actor event, if any."""
        with self._lock:
            return self._actor_events[-1] if self._actor_events else None

    @property
    def actor_control_latest_frame(self) -> ActorControlFrameV3 | None:
        """Return the latest internal actor control frame, if any."""
        with self._lock:
            return self._actor_control_frames[-1] if self._actor_control_frames else None

    def set_actor_context_provider(
        self,
        provider: Callable[[], ActorEventContext | dict[str, object]] | None,
    ) -> None:
        """Attach or replace the actor event context provider."""
        with self._lock:
            self._actor_context_provider = provider

    def set_actor_trace_writer(self, writer: ActorTraceWriter | None) -> None:
        """Attach or replace the optional actor trace writer."""
        with self._lock:
            self._actor_trace_writer = writer

    def set_performance_episode_writer(self, writer: PerformanceEpisodeV3Writer | None) -> None:
        """Attach or replace the optional performance episode writer."""
        with self._lock:
            self._performance_episode_writer = writer

    def set_discourse_episode_collector(self, collector: Any | None) -> None:
        """Attach or replace the optional internal discourse episode collector."""
        with self._lock:
            self._discourse_episode_collector = collector

    def set_actor_control_scheduler(self, scheduler: ActorControlScheduler | None) -> None:
        """Attach or replace the optional internal actor control scheduler."""
        with self._lock:
            self._actor_control_scheduler = scheduler

    def emit(
        self,
        *,
        event_type: str,
        source: str,
        mode: BrowserInteractionMode | str,
        session_id: object = None,
        client_id: object = None,
        metadata: object = None,
        reason_codes: object = None,
    ) -> BrowserPerformanceEvent:
        """Emit one public-safe event and return it."""
        try:
            if isinstance(mode, BrowserInteractionMode):
                resolved_mode = mode
            else:
                resolved_mode = BrowserInteractionMode(str(mode))
        except ValueError:
            resolved_mode = BrowserInteractionMode.WAITING
        actor_event: ActorEventV2 | None = None
        actor_trace_writer: ActorTraceWriter | None = None
        performance_episode_writer: PerformanceEpisodeV3Writer | None = None
        discourse_episode_collector: Any | None = None
        actor_control_scheduler: ActorControlScheduler | None = None
        with self._lock:
            event = BrowserPerformanceEvent(
                event_id=self._next_event_id,
                event_type=_safe_token(event_type, default="runtime.event", limit=96),
                source=_safe_token(source, default="runtime", limit=64),
                mode=resolved_mode,
                session_id=_safe_optional_token(session_id, limit=96),
                client_id=_safe_optional_token(client_id, limit=96),
                metadata=sanitize_metadata(metadata),
                reason_codes=_safe_reason_codes(reason_codes),
            )
            self._next_event_id += 1
            self._events.append(event)
            context: ActorEventContext | dict[str, object] | None = None
            if self._actor_context_provider is not None:
                try:
                    context = self._actor_context_provider()
                except Exception:
                    context = ActorEventContext()
            actor_event = actor_event_from_performance_event(event, context=context)
            self._actor_events.append(actor_event)
            actor_trace_writer = self._actor_trace_writer
            performance_episode_writer = self._performance_episode_writer
            discourse_episode_collector = self._discourse_episode_collector
            actor_control_scheduler = self._actor_control_scheduler
        if actor_trace_writer is not None and actor_event is not None:
            try:
                actor_trace_writer.append(actor_event)
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                logger.warning("Suppressed actor trace writer failure: %s", type(exc).__name__)
        if performance_episode_writer is not None and actor_event is not None:
            try:
                performance_episode_writer.append(
                    actor_event,
                    terminal_event_type=event.event_type,
                )
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                logger.warning(
                    "Suppressed performance episode writer failure: %s",
                    type(exc).__name__,
                )
        if discourse_episode_collector is not None and actor_event is not None:
            try:
                discourse_episode_collector.append(
                    actor_event,
                    terminal_event_type=event.event_type,
                )
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                logger.warning(
                    "Suppressed discourse episode collector failure: %s",
                    type(exc).__name__,
                )
        if actor_control_scheduler is not None and actor_event is not None:
            try:
                control_frame = actor_control_scheduler.observe_actor_event(actor_event)
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                logger.warning(
                    "Suppressed actor control scheduler failure: %s",
                    type(exc).__name__,
                )
                control_frame = None
            if control_frame is not None:
                with self._lock:
                    self._actor_control_frames.append(control_frame)
        return event

    def recent(self, *, after_id: int = 0, limit: int = 50) -> list[BrowserPerformanceEvent]:
        """Return recent events newer than ``after_id`` in ascending order."""
        bounded_limit = max(0, min(int(limit), 200))
        if bounded_limit <= 0:
            return []
        with self._lock:
            events = [event for event in self._events if event.event_id > int(after_id)]
            return events[-bounded_limit:]

    def as_payload(self, *, after_id: int = 0, limit: int = 50) -> dict[str, object]:
        """Return a polling payload for recent browser performance events."""
        events = self.recent(after_id=after_id, limit=limit)
        return {
            "schema_version": 1,
            "available": True,
            "after_id": max(0, int(after_id)),
            "limit": max(0, min(int(limit), 200)),
            "latest_event_id": self.latest_event_id,
            "events": [event.as_dict() for event in events],
        }

    def actor_recent(self, *, after_id: int = 0, limit: int = 50) -> list[ActorEventV2]:
        """Return recent actor events newer than ``after_id`` in ascending order."""
        bounded_limit = max(0, min(int(limit), 200))
        if bounded_limit <= 0:
            return []
        with self._lock:
            events = [event for event in self._actor_events if event.event_id > int(after_id)]
            return events[-bounded_limit:]

    def actor_payload(self, *, after_id: int = 0, limit: int = 50) -> dict[str, object]:
        """Return a polling payload for recent actor events."""
        events = self.actor_recent(after_id=after_id, limit=limit)
        return {
            "schema_version": 2,
            "available": True,
            "after_id": max(0, int(after_id)),
            "limit": max(0, min(int(limit), 200)),
            "latest_event_id": self.actor_latest_event_id,
            "events": [event.as_dict() for event in events],
        }

    def actor_control_recent(
        self,
        *,
        after_sequence: int = 0,
        limit: int = 50,
    ) -> list[ActorControlFrameV3]:
        """Return recent internal control frames newer than ``after_sequence``."""
        bounded_limit = max(0, min(int(limit), 200))
        if bounded_limit <= 0:
            return []
        with self._lock:
            frames = [
                frame
                for frame in self._actor_control_frames
                if frame.sequence > int(after_sequence)
            ]
            return frames[-bounded_limit:]

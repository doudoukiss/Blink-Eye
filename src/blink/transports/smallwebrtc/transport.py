#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Small WebRTC transport implementation for Blink.

This module provides a WebRTC transport implementation using aiortc for
real-time audio and video communication. It supports bidirectional media
streaming, application messaging, and client connection management.
"""

import asyncio
import fractions
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from blink.frames.frames import (
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    SpriteFrame,
    StartFrame,
    UserImageRawFrame,
    UserImageRequestFrame,
)
from blink.processors.frame_processor import FrameDirection
from blink.transports.base_input import BaseInputTransport
from blink.transports.base_output import BaseOutputTransport
from blink.transports.base_transport import BaseTransport, TransportParams
from blink.transports.smallwebrtc.connection import SmallWebRTCConnection

try:
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
    from av import AudioFrame, AudioResampler, VideoFrame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the SmallWebRTC, you need to `pip install blink-ai[webrtc]`.")
    raise Exception(f"Missing module: {e}")

CAM_VIDEO_SOURCE = "camera"
SCREEN_VIDEO_SOURCE = "screenVideo"
MIC_AUDIO_SOURCE = "microphone"
_TRACK_TIMEOUT_SECS = 2.0
_STALL_REMINDER_SECS = 30.0
_AUDIO_OUTPUT_WRITE_FAILURE_REMINDER_SECS = 10.0
_AUDIO_OUTPUT_WRITE_TIMEOUT_GRACE_SECS = 2.0
_AUDIO_OUTPUT_WRITE_TIMEOUT_MAX_SECS = 30.0
_DEFAULT_INPUT_VIDEO_FRAMERATE = 1


@dataclass
class _TrackStallState:
    """Runtime state for rate-limited track stall warnings."""

    track_id: int | None = None
    enabled: bool = False
    active_reason: str | None = None
    last_warning_monotonic: float = 0.0
    consecutive_failures: int = 0
    last_frame_monotonic: float | None = None
    stall_notified: bool = False


@dataclass(frozen=True)
class TrackHealthEvent:
    """Structured media-track health event emitted by the transport."""

    source: str
    reason: str
    consecutive_failures: int
    last_frame_age_ms: int | None
    enabled: bool


class SmallWebRTCCallbacks(BaseModel):
    """Callback handlers for SmallWebRTC events.

    Parameters:
        on_app_message: Called when an application message is received.
        on_client_connected: Called when a client establishes connection.
        on_client_disconnected: Called when a client disconnects.
    """

    on_app_message: Callable[[Any, str], Awaitable[None]]
    on_client_connected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_client_disconnected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_video_track_stalled: Callable[[TrackHealthEvent], Awaitable[None]]
    on_video_track_resumed: Callable[[TrackHealthEvent], Awaitable[None]]
    on_audio_track_stalled: Callable[[TrackHealthEvent], Awaitable[None]]
    on_audio_track_resumed: Callable[[TrackHealthEvent], Awaitable[None]]


class RawAudioTrack(AudioStreamTrack):
    """Custom audio stream track for WebRTC output.

    Handles audio frame generation and timing for WebRTC transmission,
    supporting queued audio data with proper synchronization.
    """

    def __init__(self, sample_rate: int, auto_silence: bool = True):
        """Initialize the raw audio track.

        Args:
            sample_rate: The audio sample rate in Hz.
            auto_silence: If True, emit silence when the queue is empty. If False,
                wait until audio data is available.
        """
        super().__init__()
        self._sample_rate = sample_rate
        self._auto_silence = auto_silence
        self._samples_per_10ms = sample_rate * 10 // 1000
        self._bytes_per_10ms = self._samples_per_10ms * 2  # 16-bit (2 bytes per sample)
        self._timestamp = 0
        self._start = time.time()
        # Queue of (bytes, future), broken into 10ms sub chunks as needed
        self._chunk_queue = deque()

    def add_audio_bytes(self, audio_bytes: bytes):
        """Add audio bytes to the buffer for transmission.

        Args:
            audio_bytes: Raw audio data to queue for transmission.

        Returns:
            A Future that completes when the data is processed.

        Raises:
            ValueError: If audio bytes are not a multiple of 10ms size.
        """
        if len(audio_bytes) % self._bytes_per_10ms != 0:
            raise ValueError("Audio bytes must be a multiple of 10ms size.")
        future = asyncio.get_running_loop().create_future()

        # Break input into 10ms chunks
        for i in range(0, len(audio_bytes), self._bytes_per_10ms):
            chunk = audio_bytes[i : i + self._bytes_per_10ms]
            # Only the last chunk carries the future to be resolved once fully consumed
            fut = future if i + self._bytes_per_10ms >= len(audio_bytes) else None
            self._chunk_queue.append((chunk, fut))

        return future

    def clear_pending_audio(self) -> int:
        """Clear queued audio that has not been consumed by the WebRTC sender."""
        cleared = len(self._chunk_queue)
        while self._chunk_queue:
            _chunk, future = self._chunk_queue.popleft()
            if future and not future.done():
                future.cancel()
        return cleared

    async def recv(self):
        """Return the next audio frame for WebRTC transmission.

        Returns:
            An AudioFrame containing the next audio data, or silence if the queue is empty
            and ``auto_silence`` is True.
        """
        # Compute required wait time for synchronization
        if self._timestamp > 0:
            wait = self._start + (self._timestamp / self._sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        if not self._chunk_queue:
            if self._auto_silence:
                chunk = bytes(self._bytes_per_10ms)
            else:
                while not self._chunk_queue:
                    await asyncio.sleep(0.005)
                chunk, future = self._chunk_queue.popleft()
                if future and not future.done():
                    future.set_result(True)
        else:
            chunk, future = self._chunk_queue.popleft()
            if future and not future.done():
                future.set_result(True)

        # Convert the byte data to an ndarray of int16 samples
        samples = np.frombuffer(chunk, dtype=np.int16)

        # Create AudioFrame
        frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._timestamp += self._samples_per_10ms
        return frame


class RawVideoTrack(VideoStreamTrack):
    """Custom video stream track for WebRTC output.

    Handles video frame queuing and conversion for WebRTC transmission.
    """

    def __init__(self, width, height):
        """Initialize the raw video track.

        Args:
            width: Video frame width in pixels.
            height: Video frame height in pixels.
        """
        super().__init__()
        self._width = width
        self._height = height
        self._video_buffer = asyncio.Queue()

    def add_video_frame(self, frame):
        """Add a video frame to the transmission buffer.

        Args:
            frame: The video frame to queue for transmission.
        """
        self._video_buffer.put_nowait(frame)

    async def recv(self):
        """Return the next video frame for WebRTC transmission.

        Returns:
            A VideoFrame ready for WebRTC transmission.
        """
        raw_frame = await self._video_buffer.get()

        # Convert bytes to NumPy array
        frame_data = np.frombuffer(raw_frame.image, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )

        frame = VideoFrame.from_ndarray(frame_data, format="rgb24")

        # Assign timestamp
        frame.pts, frame.time_base = await self.next_timestamp()

        return frame


class SmallWebRTCClient:
    """WebRTC client implementation for handling connections and media streams.

    Manages WebRTC peer connections, audio/video streaming, and application
    messaging through the SmallWebRTCConnection interface.
    """

    def __init__(self, webrtc_connection: SmallWebRTCConnection, callbacks: SmallWebRTCCallbacks):
        """Initialize the WebRTC client.

        Args:
            webrtc_connection: The underlying WebRTC connection handler.
            callbacks: Event callbacks for connection and message handling.
        """
        self._webrtc_connection = webrtc_connection
        self._closing = False
        self._callbacks = callbacks

        self._audio_output_track = None
        self._video_output_track = None
        self._audio_input_track: Optional[AudioStreamTrack] = None
        self._video_input_track: Optional[VideoStreamTrack] = None
        self._screen_video_track: Optional[VideoStreamTrack] = None
        self._audio_output_write_failure_count = 0
        self._last_audio_output_write_failure_reason = None
        self._last_audio_output_write_failure_warning_monotonic = 0.0

        self._params = None
        self._audio_in_channels = None
        self._in_sample_rate = None
        self._out_sample_rate = None
        self._leave_counter = 0
        self._track_stall_states = self._create_track_stall_states()
        self._last_video_emit_monotonic: dict[str, float] = {}

        # Audio resampler - will be configured during setup with target sample rate
        self._audio_in_resampler = None

        @self._webrtc_connection.event_handler("connected")
        async def on_connected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection established.")
            await self._handle_client_connected()

        @self._webrtc_connection.event_handler("disconnected")
        async def on_disconnected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection lost.")
            await self._handle_peer_disconnected()

        @self._webrtc_connection.event_handler("closed")
        async def on_closed(connection: SmallWebRTCConnection):
            logger.debug("Client connection closed.")
            await self._handle_client_closed()

        @self._webrtc_connection.event_handler("app-message")
        async def on_app_message(connection: SmallWebRTCConnection, message: Any):
            await self._handle_app_message(message, connection.pc_id)

    def _create_track_stall_states(self) -> dict[str, _TrackStallState]:
        """Create the per-source stall state used for warning throttling."""
        return {
            MIC_AUDIO_SOURCE: _TrackStallState(),
            CAM_VIDEO_SOURCE: _TrackStallState(),
            SCREEN_VIDEO_SOURCE: _TrackStallState(),
        }

    def _track_source_label(self, source: str) -> str:
        """Return a human-readable label for a transport media source."""
        labels = {
            MIC_AUDIO_SOURCE: "microphone",
            CAM_VIDEO_SOURCE: "camera",
            SCREEN_VIDEO_SOURCE: "screen share",
        }
        return labels.get(source, source)

    def _track_enabled(self, track: Any) -> bool:
        """Return the current enabled state for a media track."""
        if track is None:
            return False

        is_enabled = getattr(track, "is_enabled", None)
        if callable(is_enabled):
            return bool(is_enabled())

        return True

    def _sync_track_stall_state(self, source: str, track: Any) -> _TrackStallState:
        """Reset stall state when a track changes, disconnects, or toggles state."""
        state = self._track_stall_states[source]
        track_id = id(track) if track is not None else None
        enabled = self._track_enabled(track)

        if state.track_id != track_id or state.enabled != enabled:
            state.active_reason = None
            state.last_warning_monotonic = 0.0
            state.consecutive_failures = 0
            state.last_frame_monotonic = None
            state.stall_notified = False

        state.track_id = track_id
        state.enabled = enabled
        return state

    def _build_track_health_event(self, source: str, reason: str) -> TrackHealthEvent:
        """Build a structured health event from the current stall state."""
        state = self._track_stall_states[source]
        frame_age_ms = (
            max(0, int((time.monotonic() - state.last_frame_monotonic) * 1000))
            if state.last_frame_monotonic is not None
            else None
        )
        return TrackHealthEvent(
            source=source,
            reason=reason,
            consecutive_failures=state.consecutive_failures,
            last_frame_age_ms=frame_age_ms,
            enabled=state.enabled,
        )

    async def _notify_track_stalled(self, source: str, *, reason: str) -> None:
        """Emit one structured transport stall event once until frames resume."""
        state = self._track_stall_states[source]
        if state.stall_notified:
            return
        state.stall_notified = True
        event = self._build_track_health_event(source, reason)
        if source in {CAM_VIDEO_SOURCE, SCREEN_VIDEO_SOURCE}:
            await self._callbacks.on_video_track_stalled(event)
        elif source == MIC_AUDIO_SOURCE:
            await self._callbacks.on_audio_track_stalled(event)

    async def _notify_track_resumed(self, source: str, *, reason: str = "frames_resumed") -> None:
        """Emit one structured transport resume event after a stalled track recovers."""
        state = self._track_stall_states[source]
        if not state.stall_notified:
            return
        event = self._build_track_health_event(source, reason)
        if source in {CAM_VIDEO_SOURCE, SCREEN_VIDEO_SOURCE}:
            await self._callbacks.on_video_track_resumed(event)
        elif source == MIC_AUDIO_SOURCE:
            await self._callbacks.on_audio_track_resumed(event)

    def _mark_track_failure(self, source: str, *, reason: str) -> _TrackStallState:
        """Increment failure counters for a stalled track read."""
        state = self._track_stall_states[source]
        state.active_reason = reason
        state.consecutive_failures += 1
        return state

    def _clear_track_stall(self, source: str, track: Any) -> bool:
        """Clear the active stall state after frames resume."""
        state = self._sync_track_stall_state(source, track)
        resumed = state.stall_notified
        state.active_reason = None
        state.last_warning_monotonic = 0.0
        state.consecutive_failures = 0
        state.last_frame_monotonic = time.monotonic()
        state.stall_notified = False
        return resumed

    def _warn_track_stall(self, source: str, *, media_kind: str, reason: str) -> None:
        """Emit rate-limited warnings for repeated stalled media reads."""
        state = self._track_stall_states[source]
        now = time.monotonic()
        first_warning = state.active_reason != reason or state.last_warning_monotonic == 0.0

        if not first_warning and now - state.last_warning_monotonic < _STALL_REMINDER_SECS:
            return

        source_label = self._track_source_label(source)
        if reason == "timeout":
            message = (
                f"No {media_kind} frame received from {source_label} within "
                f"{_TRACK_TIMEOUT_SECS:.1f}s; suppressing repeated warnings until frames resume."
                if first_warning
                else f"Still not receiving {media_kind} frames from {source_label}; "
                "suppressing repeated warnings until frames resume."
            )
        else:
            message = (
                f"Media stream error while reading {media_kind} from {source_label}; "
                "suppressing repeated warnings until frames resume."
                if first_warning
                else f"Still seeing media stream errors while reading {media_kind} "
                f"from {source_label}; suppressing repeated warnings until frames resume."
            )

        logger.warning(message)
        state.active_reason = reason
        state.last_warning_monotonic = now

    def _convert_frame(self, frame: VideoFrame) -> np.ndarray:
        """Convert a video frame to RGB24 using PyAV's built-in conversion.

        Args:
            frame: The input video frame.

        Returns:
            The converted RGB frame as a NumPy array.
        """
        return frame.to_ndarray(format="rgb24")

    def _video_frame_interval_secs(self, framerate: int | None) -> float:
        """Return the minimum seconds between emitted input video frames."""
        if framerate is None or framerate <= 0:
            return 0.0
        return 1.0 / float(framerate)

    def _video_frame_due(self, video_source: str, framerate: int | None) -> bool:
        """Return True when the source should emit a converted video frame now."""
        interval_secs = self._video_frame_interval_secs(framerate)
        if interval_secs <= 0:
            return True

        now = time.monotonic()
        previous = self._last_video_emit_monotonic.get(video_source)
        if previous is not None and now - previous < interval_secs:
            return False

        self._last_video_emit_monotonic[video_source] = now
        return True

    async def read_video_frame(self, video_source: str, *, framerate: int = 0):
        """Read video frames from the WebRTC connection.

        Reads a video frame from the given MediaStreamTrack, converts it to RGB,
        and creates an InputImageRawFrame.

        Args:
            video_source: Video source to capture ("camera" or "screenVideo").
            framerate: Maximum emitted frame rate. A value <= 0 emits every frame.

        Yields:
            UserImageRawFrame objects containing video data from the peer.
        """
        while True:
            video_track = (
                self._video_input_track
                if video_source == CAM_VIDEO_SOURCE
                else self._screen_video_track
            )
            self._sync_track_stall_state(video_source, video_track)
            if video_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(video_track.recv(), timeout=_TRACK_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                if (
                    self._webrtc_connection.is_connected()
                    and video_track
                    and video_track.is_enabled()
                ):
                    self._mark_track_failure(video_source, reason="timeout")
                    self._warn_track_stall(video_source, media_kind="video", reason="timeout")
                    if self._track_stall_states[video_source].consecutive_failures >= 2:
                        await self._notify_track_stalled(video_source, reason="timeout")
                frame = None
            except MediaStreamError:
                self._mark_track_failure(video_source, reason="media-error")
                self._warn_track_stall(video_source, media_kind="video", reason="media-error")
                if self._track_stall_states[video_source].consecutive_failures >= 2:
                    await self._notify_track_stalled(video_source, reason="media-error")
                frame = None

            if frame is None or not isinstance(frame, VideoFrame):
                # If no valid frame, sleep for a bit
                await asyncio.sleep(0.01)
                continue

            resumed = self._clear_track_stall(video_source, video_track)
            if resumed:
                await self._notify_track_resumed(video_source)

            if not self._video_frame_due(video_source, framerate):
                del frame
                await asyncio.sleep(0)
                continue

            frame_rgb = self._convert_frame(frame)
            image_bytes = frame_rgb.tobytes()
            del frame_rgb  # free RGB array immediately

            image_frame = UserImageRawFrame(
                user_id=self._webrtc_connection.pc_id,
                image=image_bytes,
                size=(frame.width, frame.height),
                format="RGB",
            )
            image_frame.transport_source = video_source
            image_frame.pts = frame.pts

            del frame  # free original VideoFrame
            del image_bytes  # reference kept in image_frame

            yield image_frame

    async def read_audio_frame(self):
        """Read audio frames from the WebRTC connection.

        Reads 20ms of audio from the given MediaStreamTrack and creates an InputAudioRawFrame.

        Yields:
            InputAudioRawFrame objects containing audio data from the peer.
        """
        while True:
            self._sync_track_stall_state(MIC_AUDIO_SOURCE, self._audio_input_track)
            if self._audio_input_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(
                    self._audio_input_track.recv(),
                    timeout=_TRACK_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                if (
                    self._webrtc_connection.is_connected()
                    and self._audio_input_track
                    and self._audio_input_track.is_enabled()
                ):
                    self._mark_track_failure(MIC_AUDIO_SOURCE, reason="timeout")
                    self._warn_track_stall(
                        MIC_AUDIO_SOURCE,
                        media_kind="audio",
                        reason="timeout",
                    )
                    if self._track_stall_states[MIC_AUDIO_SOURCE].consecutive_failures >= 2:
                        await self._notify_track_stalled(MIC_AUDIO_SOURCE, reason="timeout")
                frame = None
            except MediaStreamError:
                self._mark_track_failure(MIC_AUDIO_SOURCE, reason="media-error")
                self._warn_track_stall(
                    MIC_AUDIO_SOURCE,
                    media_kind="audio",
                    reason="media-error",
                )
                if self._track_stall_states[MIC_AUDIO_SOURCE].consecutive_failures >= 2:
                    await self._notify_track_stalled(MIC_AUDIO_SOURCE, reason="media-error")
                frame = None

            if frame is None or not isinstance(frame, AudioFrame):
                # If we don't read any audio let's sleep for a little bit (i.e. busy wait).
                await asyncio.sleep(0.01)
                continue

            resumed = self._clear_track_stall(MIC_AUDIO_SOURCE, self._audio_input_track)
            if resumed:
                await self._notify_track_resumed(MIC_AUDIO_SOURCE)

            # Resample if needed, otherwise use the frame as-is
            frames_to_process = (
                self._audio_in_resampler.resample(frame)
                if frame.sample_rate != self._in_sample_rate
                else [frame]
            )

            for processed_frame in frames_to_process:
                # Convert to 16-bit PCM bytes
                pcm_array = processed_frame.to_ndarray().astype(np.int16)
                pcm_bytes = pcm_array.tobytes()
                del pcm_array  # free NumPy array immediately

                audio_frame = InputAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=self._in_sample_rate,
                    num_channels=self._audio_in_channels,
                )
                audio_frame.pts = frame.pts
                del pcm_bytes  # reference kept in audio_frame

                yield audio_frame

            del frame  # free original AudioFrame

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the WebRTC connection.

        Args:
            frame: The audio frame to transmit.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        if self._can_send_media() and self._audio_output_track:
            future = self._audio_output_track.add_audio_bytes(frame.audio)
            try:
                await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=self._audio_output_write_timeout_secs(frame),
                )
            except TimeoutError:
                self._audio_output_track.clear_pending_audio()
                self._record_audio_output_write_failure("audio_output_not_consumed")
                return False
            return True
        self._record_audio_output_write_failure(self._audio_output_write_failure_reason())
        return False

    def _audio_output_write_timeout_secs(self, frame: OutputAudioRawFrame) -> float:
        sample_rate = int(frame.sample_rate or self._out_sample_rate or 0)
        if sample_rate <= 0:
            return _AUDIO_OUTPUT_WRITE_TIMEOUT_GRACE_SECS
        channels = max(1, int(frame.num_channels or 1))
        duration_secs = len(frame.audio) / float(sample_rate * channels * 2)
        return max(
            _AUDIO_OUTPUT_WRITE_TIMEOUT_GRACE_SECS,
            min(
                _AUDIO_OUTPUT_WRITE_TIMEOUT_MAX_SECS,
                duration_secs + _AUDIO_OUTPUT_WRITE_TIMEOUT_GRACE_SECS,
            ),
        )

    async def interrupt_audio_output(self):
        """Clear queued browser audio after an interruption."""
        if self._audio_output_track is not None:
            self._audio_output_track.clear_pending_audio()

    def _audio_output_write_failure_reason(self) -> str:
        if not self._can_send_media():
            return "webrtc_disconnected"
        if self._audio_output_track is None:
            return "missing_audio_output_track"
        return "audio_output_unavailable"

    def _record_audio_output_write_failure(self, reason: str) -> None:
        self._audio_output_write_failure_count += 1
        self._last_audio_output_write_failure_reason = reason
        now = time.monotonic()
        if (
            now - self._last_audio_output_write_failure_warning_monotonic
            < _AUDIO_OUTPUT_WRITE_FAILURE_REMINDER_SECS
        ):
            return
        self._last_audio_output_write_failure_warning_monotonic = now
        logger.warning("Small WebRTC output audio write skipped: {}", reason)

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the WebRTC connection.

        Args:
            frame: The video frame to transmit.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        if self._can_send_media() and self._video_output_track:
            self._video_output_track.add_video_frame(frame)
            return True
        return False

    async def setup(self, _params: TransportParams, frame):
        """Set up the client with transport parameters.

        Args:
            _params: Transport configuration parameters.
            frame: The initialization frame containing setup data.
        """
        self._audio_in_channels = _params.audio_in_channels
        self._in_sample_rate = _params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = _params.audio_out_sample_rate or frame.audio_out_sample_rate
        self._params = _params
        self._leave_counter += 1
        self._audio_in_resampler = AudioResampler("s16", "mono", self._in_sample_rate)

    async def connect(self):
        """Establish the WebRTC connection."""
        if self._webrtc_connection.is_connected():
            # already initialized
            return

        logger.info(f"Connecting to Small WebRTC")
        await self._webrtc_connection.connect()

    async def disconnect(self):
        """Disconnect from the WebRTC peer."""
        self._leave_counter -= 1
        if self._leave_counter > 0:
            return

        if self.is_connected and not self.is_closing:
            logger.info(f"Disconnecting to Small WebRTC")
            self._closing = True
            await self._webrtc_connection.disconnect()
            await self._handle_peer_disconnected()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send an application message through the WebRTC connection.

        Args:
            frame: The message frame to send.
        """
        if self._can_send():
            self._webrtc_connection.send_app_message(frame.message)

    async def _handle_client_connected(self):
        """Handle client connection establishment."""
        # There is nothing to do here yet, the pipeline is still not ready
        if not self._params:
            return

        self._audio_input_track = self._webrtc_connection.audio_input_track()
        self._video_input_track = self._webrtc_connection.video_input_track()
        self._screen_video_track = self._webrtc_connection.screen_video_input_track()
        if self._params.audio_out_enabled:
            self._audio_output_track = RawAudioTrack(
                sample_rate=self._out_sample_rate,
                auto_silence=self._params.audio_out_auto_silence,
            )
            if self._webrtc_connection.replace_audio_track(self._audio_output_track) is False:
                self._audio_output_track = None

        if self._params.video_out_enabled:
            self._video_output_track = RawVideoTrack(
                width=self._params.video_out_width, height=self._params.video_out_height
            )
            if self._webrtc_connection.replace_video_track(self._video_output_track) is False:
                self._video_output_track = None

        await self._callbacks.on_client_connected(self._webrtc_connection)

    async def _handle_peer_disconnected(self):
        """Handle peer disconnection cleanup."""
        self._track_stall_states = self._create_track_stall_states()
        self._last_video_emit_monotonic.clear()
        self._audio_input_track = None
        self._video_input_track = None
        self._screen_video_track = None
        self._audio_output_track = None
        self._video_output_track = None

    async def _handle_client_closed(self):
        """Handle client connection closure."""
        self._track_stall_states = self._create_track_stall_states()
        self._last_video_emit_monotonic.clear()
        self._audio_input_track = None
        self._video_input_track = None
        self._screen_video_track = None
        self._audio_output_track = None
        self._video_output_track = None

        # Trigger `on_client_disconnected` if the client actually disconnects,
        # that is, we are not the ones disconnecting.
        if not self._closing:
            await self._callbacks.on_client_disconnected(self._webrtc_connection)

    async def _handle_app_message(self, message: Any, sender: str):
        """Handle incoming application messages."""
        await self._callbacks.on_app_message(message, sender)

    def _can_send(self):
        """Check if the connection is ready for sending data."""
        return self.is_connected and not self.is_closing

    def _can_send_media(self):
        """Check if media sender tracks can still be written to."""
        media_ready = getattr(self._webrtc_connection, "is_media_transport_ready", None)
        if callable(media_ready):
            return bool(media_ready()) and not self.is_closing
        return self._can_send()

    @property
    def is_connected(self) -> bool:
        """Check if the WebRTC connection is established.

        Returns:
            True if connected to the peer.
        """
        return self._webrtc_connection.is_connected()

    @property
    def is_closing(self) -> bool:
        """Check if the connection is in the process of closing.

        Returns:
            True if the connection is closing.
        """
        return self._closing


class SmallWebRTCInputTransport(BaseInputTransport):
    """Input transport implementation for SmallWebRTC.

    Handles incoming audio and video streams from WebRTC peers,
    including user image requests and application message handling.
    """

    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the WebRTC input transport.

        Args:
            client: The WebRTC client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None
        self._receive_video_task = None
        self._receive_screen_video_task = None
        self._camera_framerate = _DEFAULT_INPUT_VIDEO_FRAMERATE
        self._screen_video_framerate = _DEFAULT_INPUT_VIDEO_FRAMERATE
        self._image_requests: List[UserImageRequestFrame] = []

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames including user image requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, UserImageRequestFrame):
            await self.request_participant_image(frame)

    async def start(self, frame: StartFrame):
        """Start the input transport and establish WebRTC connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(self._params, frame)
        await self._client.connect()
        await self.set_transport_ready(frame)
        if not self._receive_audio_task and self._params.audio_in_enabled:
            self._receive_audio_task = self.create_task(self._receive_audio())
        if not self._receive_video_task and self._params.video_in_enabled:
            self._receive_video_task = self.create_task(
                self._receive_video(CAM_VIDEO_SOURCE, framerate=self._camera_framerate)
            )

    async def _stop_tasks(self):
        """Stop all background tasks."""
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        if self._receive_video_task:
            await self.cancel_task(self._receive_video_task)
            self._receive_video_task = None

    async def stop(self, frame: EndFrame):
        """Stop the input transport and disconnect from WebRTC.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport and disconnect immediately.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def _receive_audio(self):
        """Background task for receiving audio frames from WebRTC."""
        try:
            audio_iterator = self._client.read_audio_frame()
            async for audio_frame in audio_iterator:
                if audio_frame:
                    await self.push_audio_frame(audio_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def _receive_video(self, video_source: str, *, framerate: int = 0):
        """Background task for receiving video frames from WebRTC.

        Args:
            video_source: Video source to capture ("camera" or "screenVideo").
            framerate: Maximum emitted frame rate. A value <= 0 emits every frame.
        """
        try:
            video_iterator = self._client.read_video_frame(video_source, framerate=framerate)
            async for video_frame in video_iterator:
                if video_frame:
                    await self.push_video_frame(video_frame)

                    # Check if there are any pending image requests and create
                    # UserImageRawFrame. Use a shallow copy so we can remove
                    # elements.
                    for request_frame in self._image_requests[:]:
                        request_text = request_frame.text if request_frame else None
                        add_to_context = request_frame.append_to_context if request_frame else None
                        if request_frame.video_source == video_source:
                            # Create UserImageRawFrame using the current video frame
                            image_frame = UserImageRawFrame(
                                user_id=request_frame.user_id,
                                image=video_frame.image,
                                size=video_frame.size,
                                format=video_frame.format,
                                text=request_text,
                                append_to_context=add_to_context,
                                request=request_frame,
                            )
                            image_frame.transport_source = video_source
                            # Push the frame to the pipeline
                            await self.push_video_frame(image_frame)
                            # Remove from pending requests
                            self._image_requests.remove(request_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def push_app_message(self, message: Any):
        """Push an application message into the pipeline.

        Args:
            message: The application message to process.
        """
        logger.debug(f"Received app message inside SmallWebRTCInputTransport  {message}")
        await self.broadcast_frame(InputTransportMessageFrame, message=message)

    # Add this method similar to DailyInputTransport.request_participant_image
    async def request_participant_image(self, frame: UserImageRequestFrame):
        """Request an image frame from the participant's video stream.

        When a UserImageRequestFrame is received, this method will store the request
        and the next video frame received will be converted to a UserImageRawFrame.

        Args:
            frame: The user image request frame.
        """
        logger.debug(f"Requesting image from participant: {frame.user_id}")

        # Store the request
        self._image_requests.append(frame)

        # Default to camera if no source specified
        if frame.video_source is None:
            frame.video_source = CAM_VIDEO_SOURCE
        # If we're not already receiving video, try to get a frame now
        if (
            frame.video_source == CAM_VIDEO_SOURCE
            and not self._receive_video_task
            and self._params.video_in_enabled
        ):
            # Start video reception if it's not already running
            self._receive_video_task = self.create_task(
                self._receive_video(CAM_VIDEO_SOURCE, framerate=self._camera_framerate)
            )
        elif (
            frame.video_source == SCREEN_VIDEO_SOURCE
            and not self._receive_screen_video_task
            and self._params.video_in_enabled
        ):
            # Start screen video reception if it's not already running
            self._receive_screen_video_task = self.create_task(
                self._receive_video(SCREEN_VIDEO_SOURCE, framerate=self._screen_video_framerate)
            )

    async def capture_participant_media(
        self,
        source: str = CAM_VIDEO_SOURCE,
        framerate: int = 0,
    ):
        """Capture media from a specific participant.

        Args:
            source: Media source to capture from. ("camera", "microphone", or "screenVideo")
            framerate: Maximum emitted video frame rate. A value <= 0 uses the
                transport default for video sources.
        """
        if source == CAM_VIDEO_SOURCE and framerate > 0:
            self._camera_framerate = framerate
        elif source == SCREEN_VIDEO_SOURCE and framerate > 0:
            self._screen_video_framerate = framerate

        # If we're not already receiving video, try to get a frame now
        if (
            source == MIC_AUDIO_SOURCE
            and not self._receive_audio_task
            and self._params.audio_in_enabled
        ):
            # Start audio reception if it's not already running
            self._receive_audio_task = self.create_task(self._receive_audio())
        elif (
            source == CAM_VIDEO_SOURCE
            and not self._receive_video_task
            and self._params.video_in_enabled
        ):
            # Start video reception if it's not already running
            self._receive_video_task = self.create_task(
                self._receive_video(CAM_VIDEO_SOURCE, framerate=self._camera_framerate)
            )
        elif (
            source == SCREEN_VIDEO_SOURCE
            and not self._receive_screen_video_task
            and self._params.video_in_enabled
        ):
            # Start screen video reception if it's not already running
            self._receive_screen_video_task = self.create_task(
                self._receive_video(SCREEN_VIDEO_SOURCE, framerate=self._screen_video_framerate)
            )


class SmallWebRTCOutputTransport(BaseOutputTransport):
    """Output transport implementation for SmallWebRTC.

    Handles outgoing audio and video streams to WebRTC peers,
    including transport message sending.
    """

    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the WebRTC output transport.

        Args:
            client: The WebRTC client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

        # Whether we have seen a StartFrame already.
        self._initialized = False

    async def start(self, frame: StartFrame):
        """Start the output transport and establish WebRTC connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        if self._initialized:
            return

        self._initialized = True

        await self._client.setup(self._params, frame)
        await self._client.connect()
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport and disconnect from WebRTC.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport and disconnect immediately.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.disconnect()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a transport message through the WebRTC connection.

        Args:
            frame: The transport message frame to send.
        """
        await self._client.send_message(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write an audio frame to the WebRTC connection.

        Args:
            frame: The output audio frame to transmit.

        Returns:
            True if the audio frame was written successfully, False otherwise.
        """
        return await self._client.write_audio_frame(frame)

    async def interrupt_audio_output(self):
        """Clear queued browser audio after an interruption."""
        await self._client.interrupt_audio_output()

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write a video frame to the WebRTC connection.

        Args:
            frame: The output video frame to transmit.

        Returns:
            True if the video frame was written successfully, False otherwise.
        """
        return await self._client.write_video_frame(frame)


class SmallWebRTCTransport(BaseTransport):
    """WebRTC transport implementation for real-time communication.

    Provides bidirectional audio and video streaming over WebRTC connections
    with support for application messaging and connection event handling.

    Event handlers available:

    - on_client_connected(transport, client): Client connected to WebRTC session
    - on_client_disconnected(transport, client): Client disconnected from WebRTC session
    - on_client_message(transport, message, client): Received a data channel message
    - on_video_track_stalled(transport, event): Video track stalled
    - on_video_track_resumed(transport, event): Video track resumed
    - on_audio_track_stalled(transport, event): Audio track stalled
    - on_audio_track_resumed(transport, event): Audio track resumed

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            ...
    """

    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection,
        params: TransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        """Initialize the WebRTC transport.

        Args:
            webrtc_connection: The underlying WebRTC connection handler.
            params: Transport configuration parameters.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = SmallWebRTCCallbacks(
            on_app_message=self._on_app_message,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_video_track_stalled=self._on_video_track_stalled,
            on_video_track_resumed=self._on_video_track_resumed,
            on_audio_track_stalled=self._on_audio_track_stalled,
            on_audio_track_resumed=self._on_audio_track_resumed,
        )

        self._client = SmallWebRTCClient(webrtc_connection, self._callbacks)

        self._input: Optional[SmallWebRTCInputTransport] = None
        self._output: Optional[SmallWebRTCOutputTransport] = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_video_track_stalled")
        self._register_event_handler("on_video_track_resumed")
        self._register_event_handler("on_audio_track_stalled")
        self._register_event_handler("on_audio_track_resumed")

    def input(self) -> SmallWebRTCInputTransport:
        """Get the input transport processor.

        Returns:
            The input transport for handling incoming media streams.
        """
        if not self._input:
            self._input = SmallWebRTCInputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> SmallWebRTCOutputTransport:
        """Get the output transport processor.

        Returns:
            The output transport for handling outgoing media streams.
        """
        if not self._output:
            self._output = SmallWebRTCOutputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._output

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        """Send an image frame through the transport.

        Args:
            frame: The image frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_audio(self, frame: OutputAudioRawFrame):
        """Send an audio frame through the transport.

        Args:
            frame: The audio frame to send.
        """
        if self._output:
            await self._output.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def _on_app_message(self, message: Any, sender: str):
        """Handle incoming application messages."""
        if self._input:
            await self._input.push_app_message(message)
        await self._call_event_handler("on_app_message", message, sender)

    async def _on_client_connected(self, webrtc_connection):
        """Handle client connection events."""
        await self._call_event_handler("on_client_connected", webrtc_connection)
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_client_disconnected(self, webrtc_connection):
        """Handle client disconnection events."""
        await self._call_event_handler("on_client_disconnected", webrtc_connection)

    async def _on_video_track_stalled(self, event: TrackHealthEvent):
        """Handle video-track health degradation events."""
        await self._call_event_handler("on_video_track_stalled", event)

    async def _on_video_track_resumed(self, event: TrackHealthEvent):
        """Handle video-track recovery events."""
        await self._call_event_handler("on_video_track_resumed", event)

    async def _on_audio_track_stalled(self, event: TrackHealthEvent):
        """Handle audio-track health degradation events."""
        await self._call_event_handler("on_audio_track_stalled", event)

    async def _on_audio_track_resumed(self, event: TrackHealthEvent):
        """Handle audio-track recovery events."""
        await self._call_event_handler("on_audio_track_resumed", event)

    async def capture_participant_video(
        self,
        video_source: str = CAM_VIDEO_SOURCE,
        framerate: int = 0,
    ):
        """Capture video from a specific participant.

        Args:
            video_source: Video source to capture from ("camera" or "screenVideo").
            framerate: Maximum emitted video frame rate. A value <= 0 uses the
                SmallWebRTC default.
        """
        if self._input:
            await self._input.capture_participant_media(
                source=video_source,
                framerate=framerate,
            )

    async def capture_participant_audio(
        self,
        audio_source: str = MIC_AUDIO_SOURCE,
    ):
        """Capture audio from a specific participant.

        Args:
            audio_source: Audio source to capture from. (currently, "microphone" is the only supported option)
        """
        if self._input:
            await self._input.capture_participant_media(source=audio_source)

    async def request_renegotiation(self):
        """Request a bounded peer-connection renegotiation."""
        self._client._webrtc_connection.ask_to_renegotiate()

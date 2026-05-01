"""Feetech/STS3032 serial protocol helpers for Blink's robot head driver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterable, Protocol, Sequence

HEADER = bytes((0xFF, 0xFF))
BROADCAST_ID = 0xFE

ADDRESS_TORQUE_SWITCH = 0x28
ADDRESS_START_ACCELERATION = 0x29
ADDRESS_TARGET_POSITION = 0x2A
ADDRESS_RUNNING_SPEED = 0x2E
ADDRESS_PRESENT_POSITION = 0x38
ADDRESS_PRESENT_SPEED = 0x3A
ADDRESS_PRESENT_LOAD = 0x3C
ADDRESS_PRESENT_VOLTAGE = 0x3E
ADDRESS_PRESENT_TEMPERATURE = 0x3F
ADDRESS_PRESENT_ASYNC_FLAG = 0x40
ADDRESS_PRESENT_STATUS = 0x41
ADDRESS_PRESENT_MOVING = 0x42
ADDRESS_PRESENT_CURRENT = 0x45


class FeetechInstruction(IntEnum):
    """Supported Feetech instruction opcodes."""

    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    RECOVERY = 0x06
    RESET = 0x0A
    SYNC_READ = 0x82
    SYNC_WRITE = 0x83


class FeetechProtocolError(ValueError):
    """Raised when a Feetech frame is malformed."""


class FeetechTransportError(RuntimeError):
    """Raised when the live serial transport cannot complete an exchange."""


class SerialConnectionProtocol(Protocol):
    """Minimal protocol required from a serial connection implementation."""

    port: str
    baudrate: int
    timeout: float

    def write(self, payload: bytes) -> int:
        """Write raw bytes to the serial bus."""

    def read(self, size: int) -> bytes:
        """Read raw bytes from the serial bus."""

    def close(self) -> None:
        """Close the serial connection."""

    def reset_input_buffer(self) -> None:
        """Discard unread bytes from the receive buffer."""

    def reset_output_buffer(self) -> None:
        """Discard buffered outgoing bytes."""


@dataclass(frozen=True)
class FeetechPacket:
    """Decoded Feetech instruction packet."""

    servo_id: int
    length: int
    instruction: int
    parameters: bytes
    checksum: int
    raw_frame: bytes


@dataclass(frozen=True)
class FeetechStatusPacket:
    """Decoded Feetech status packet."""

    servo_id: int
    length: int
    error: int
    parameters: bytes
    checksum: int
    raw_frame: bytes


def normalize_servo_id(servo_id: int) -> int:
    """Validate and normalize a servo ID."""
    normalized = int(servo_id)
    if not 0 <= normalized <= 0xFE:
        raise FeetechProtocolError(f"invalid_servo_id:{servo_id}")
    return normalized


def checksum(payload: Iterable[int] | bytes) -> int:
    """Return the Feetech inverted byte-sum checksum."""
    total = sum(int(value) & 0xFF for value in payload) & 0xFF
    return (~total) & 0xFF


def pack_u8(value: int) -> bytes:
    """Pack an unsigned byte."""
    normalized = int(value)
    if not 0 <= normalized <= 0xFF:
        raise FeetechProtocolError(f"u8_out_of_range:{value}")
    return bytes((normalized,))


def pack_u16_le(value: int) -> bytes:
    """Pack an unsigned 16-bit little-endian value."""
    normalized = int(value)
    if not 0 <= normalized <= 0xFFFF:
        raise FeetechProtocolError(f"u16_out_of_range:{value}")
    return bytes((normalized & 0xFF, (normalized >> 8) & 0xFF))


def unpack_u16_le(payload: bytes) -> int:
    """Unpack an unsigned 16-bit little-endian value."""
    if len(payload) != 2:
        raise FeetechProtocolError(f"u16_requires_two_bytes:{len(payload)}")
    return payload[0] | (payload[1] << 8)


def encode_instruction_packet(
    servo_id: int,
    instruction: int | FeetechInstruction,
    parameters: bytes | Sequence[int] = b"",
) -> bytes:
    """Encode a Feetech instruction packet."""
    normalized_id = normalize_servo_id(servo_id)
    payload = bytes(parameters)
    length = len(payload) + 2
    body = bytes((normalized_id, length, int(instruction) & 0xFF)) + payload
    return HEADER + body + bytes((checksum(body),))


def encode_status_packet(
    servo_id: int,
    *,
    error: int = 0,
    parameters: bytes | Sequence[int] = b"",
) -> bytes:
    """Encode a Feetech status packet."""
    normalized_id = normalize_servo_id(servo_id)
    payload = bytes(parameters)
    length = len(payload) + 2
    body = bytes((normalized_id, length, int(error) & 0xFF)) + payload
    return HEADER + body + bytes((checksum(body),))


def decode_packet(frame: bytes) -> FeetechPacket:
    """Decode a Feetech instruction or status packet."""
    if len(frame) < 6:
        raise FeetechProtocolError(f"frame_too_short:{len(frame)}")
    if frame[:2] != HEADER:
        raise FeetechProtocolError("missing_header")
    servo_id = frame[2]
    length = frame[3]
    expected_size = length + 4
    if len(frame) != expected_size:
        raise FeetechProtocolError(
            f"invalid_frame_length:expected={expected_size}:actual={len(frame)}"
        )
    expected_checksum = checksum(frame[2:-1])
    if frame[-1] != expected_checksum:
        raise FeetechProtocolError(
            f"checksum_mismatch:expected=0x{expected_checksum:02X}:actual=0x{frame[-1]:02X}"
        )
    return FeetechPacket(
        servo_id=servo_id,
        length=length,
        instruction=frame[4],
        parameters=frame[5:-1],
        checksum=frame[-1],
        raw_frame=bytes(frame),
    )


def decode_status_packet(frame: bytes) -> FeetechStatusPacket:
    """Decode a Feetech status packet."""
    packet = decode_packet(frame)
    return FeetechStatusPacket(
        servo_id=packet.servo_id,
        length=packet.length,
        error=packet.instruction,
        parameters=packet.parameters,
        checksum=packet.checksum,
        raw_frame=packet.raw_frame,
    )


def ping_packet(servo_id: int) -> bytes:
    """Build a PING packet."""
    return encode_instruction_packet(servo_id, FeetechInstruction.PING)


def read_packet(servo_id: int, address: int, length: int) -> bytes:
    """Build a READ packet."""
    return encode_instruction_packet(
        servo_id,
        FeetechInstruction.READ,
        bytes((address & 0xFF, length & 0xFF)),
    )


def write_packet(servo_id: int, address: int, payload: bytes | Sequence[int]) -> bytes:
    """Build a WRITE packet."""
    return encode_instruction_packet(
        servo_id,
        FeetechInstruction.WRITE,
        bytes((address & 0xFF,)) + bytes(payload),
    )


def sync_write_packet(address: int, data_length: int, writes: Sequence[tuple[int, bytes]]) -> bytes:
    """Build a SYNC_WRITE packet."""
    parameters = bytearray((address & 0xFF, data_length & 0xFF))
    for servo_id, payload in writes:
        if len(payload) != data_length:
            raise FeetechProtocolError(
                "sync_write_payload_length_mismatch:"
                f"id={servo_id}:expected={data_length}:actual={len(payload)}"
            )
        parameters.append(normalize_servo_id(servo_id))
        parameters.extend(payload)
    return encode_instruction_packet(BROADCAST_ID, FeetechInstruction.SYNC_WRITE, parameters)


def sync_read_packet(address: int, data_length: int, servo_ids: Sequence[int]) -> bytes:
    """Build a SYNC_READ packet."""
    parameters = bytearray((address & 0xFF, data_length & 0xFF))
    for servo_id in servo_ids:
        parameters.append(normalize_servo_id(servo_id))
    return encode_instruction_packet(BROADCAST_ID, FeetechInstruction.SYNC_READ, parameters)


def build_target_position_payload(position: int, duration_ms: int = 0, speed: int = 0) -> bytes:
    """Build the 6-byte target-position payload."""
    return (
        pack_u16_le(position)
        + pack_u16_le(duration_ms)
        + pack_u16_le(speed)
    )


def format_frame_hex(frame: bytes) -> str:
    """Render a packet as a hex string."""
    return " ".join(f"{byte:02X}" for byte in frame)


def classify_open_failure(exc: Exception) -> tuple[str, str]:
    """Classify a serial-open failure into a stable reason code."""
    detail = str(exc)
    lowered = detail.lower()
    if isinstance(exc, FileNotFoundError) or "no such file" in lowered or "not found" in lowered:
        return "missing_port", detail
    if (
        "busy" in lowered
        or "resource busy" in lowered
        or "permission denied" in lowered
        or "in use" in lowered
    ):
        return "port_busy", detail
    return "serial_unavailable", detail


class LiveSerialProtocolTransport:
    """Blocking Feetech serial transport used by the live robot-head driver."""

    def __init__(
        self,
        *,
        port: str,
        baud_rate: int,
        timeout_seconds: float,
        connection_factory: Callable[[], SerialConnectionProtocol] | None = None,
    ):
        """Initialize the transport.

        Args:
            port: Serial device path.
            baud_rate: Configured baud rate.
            timeout_seconds: Read and write timeout.
            connection_factory: Optional test hook for a fake serial connection.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout_seconds = timeout_seconds
        self._connection_factory = connection_factory or self._default_connection_factory()
        self._connection: SerialConnectionProtocol | None = None

    def close(self) -> None:
        """Close the underlying serial connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def ping(self, servo_id: int) -> FeetechStatusPacket:
        """Ping a single servo ID."""
        return self._single_reply(ping_packet(servo_id), expected_response_ids=[servo_id])[0]

    def read(self, servo_id: int, address: int, length: int) -> bytes:
        """Read a register block from one servo."""
        packet = self._single_reply(
            read_packet(servo_id, address, length),
            expected_response_ids=[servo_id],
        )[0]
        if len(packet.parameters) != length:
            raise FeetechTransportError(
                f"read_length_mismatch:id={servo_id}:expected={length}:actual={len(packet.parameters)}"
            )
        return packet.parameters

    def write(self, servo_id: int, address: int, payload: bytes, *, expect_reply: bool = True) -> None:
        """Write a register block to one servo."""
        self._exchange(
            write_packet(servo_id, address, payload),
            expected_response_ids=[servo_id] if expect_reply and servo_id != BROADCAST_ID else [],
        )

    def sync_write(self, address: int, writes: Sequence[tuple[int, bytes]], *, data_length: int) -> None:
        """Write the same register block to multiple servos."""
        self._exchange(
            sync_write_packet(address, data_length, writes),
            expected_response_ids=[],
        )

    def sync_read(self, address: int, length: int, servo_ids: Sequence[int]) -> dict[int, bytes]:
        """Read the same register block from multiple servos."""
        responses = self._exchange(
            sync_read_packet(address, length, servo_ids),
            expected_response_ids=list(servo_ids),
        )
        return {packet.servo_id: packet.parameters for packet in responses}

    def set_torque(self, servo_ids: Sequence[int], *, enabled: bool) -> None:
        """Set torque state across multiple servos."""
        writes = [(servo_id, pack_u8(1 if enabled else 0)) for servo_id in servo_ids]
        self.sync_write(ADDRESS_TORQUE_SWITCH, writes, data_length=1)

    def sync_write_start_acceleration(self, payloads: Sequence[tuple[int, int]]) -> None:
        """Write start-acceleration settings to multiple servos."""
        writes = [(servo_id, pack_u8(acceleration)) for servo_id, acceleration in payloads]
        self.sync_write(ADDRESS_START_ACCELERATION, writes, data_length=1)

    def sync_write_target_positions(
        self,
        payloads: Sequence[tuple[int, int]],
        *,
        duration_ms: int,
        speed: int,
    ) -> None:
        """Write target positions to multiple servos."""
        writes = [
            (
                servo_id,
                build_target_position_payload(
                    position=position,
                    duration_ms=duration_ms,
                    speed=speed,
                ),
            )
            for servo_id, position in payloads
        ]
        self.sync_write(ADDRESS_TARGET_POSITION, writes, data_length=6)

    def _default_connection_factory(self) -> Callable[[], SerialConnectionProtocol]:
        """Create the default pyserial connection factory."""
        try:
            import serial  # type: ignore
        except ModuleNotFoundError as exc:
            raise FeetechTransportError("pyserial_not_installed") from exc

        def factory() -> SerialConnectionProtocol:
            try:
                return serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=self.timeout_seconds,
                    write_timeout=self.timeout_seconds,
                )
            except Exception as exc:
                classification, detail = classify_open_failure(exc)
                raise FeetechTransportError(f"{classification}:{detail}") from exc

        return factory

    def _single_reply(
        self,
        request_frame: bytes,
        *,
        expected_response_ids: Sequence[int],
    ) -> list[FeetechStatusPacket]:
        """Exchange one request and decode the returned status packets."""
        return self._exchange(request_frame, expected_response_ids=expected_response_ids)

    def _exchange(
        self,
        request_frame: bytes,
        *,
        expected_response_ids: Sequence[int],
    ) -> list[FeetechStatusPacket]:
        """Exchange one request frame and return the decoded replies."""
        connection = self._ensure_connection()
        try:
            connection.reset_input_buffer()
            connection.reset_output_buffer()
        except Exception:
            pass

        try:
            connection.write(request_frame)
        except Exception as exc:
            raise FeetechTransportError(f"write_failed:{exc}") from exc

        if not expected_response_ids:
            return []

        responses = [self._read_frame(connection) for _ in expected_response_ids]
        packets = [decode_status_packet(frame) for frame in responses]
        actual_ids = [packet.servo_id for packet in packets]
        if actual_ids != list(expected_response_ids):
            raise FeetechTransportError(
                f"reply_id_mismatch:expected={list(expected_response_ids)}:actual={actual_ids}"
            )
        return packets

    def _ensure_connection(self) -> SerialConnectionProtocol:
        """Return an open serial connection."""
        if self._connection is None:
            self._connection = self._connection_factory()
        return self._connection

    def _read_frame(self, connection: SerialConnectionProtocol) -> bytes:
        """Read one complete status frame from the serial stream."""
        prefix = self._read_exact(connection, 2)
        while prefix != HEADER:
            prefix = prefix[1:] + self._read_exact(connection, 1)
        header_tail = self._read_exact(connection, 2)
        servo_id = header_tail[0]
        length = header_tail[1]
        remainder = self._read_exact(connection, length)
        frame = HEADER + bytes((servo_id, length)) + remainder
        decode_status_packet(frame)
        return frame

    def _read_exact(self, connection: SerialConnectionProtocol, size: int) -> bytes:
        """Read exactly `size` bytes or raise a timeout error."""
        chunks = bytearray()
        while len(chunks) < size:
            chunk = connection.read(size - len(chunks))
            if not chunk:
                raise FeetechTransportError(
                    f"timeout:serial_timeout:expected={size}:received={len(chunks)}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

"""VIGIL Stage 1: HMAC-SHA256 authenticated camera frame packages."""

from __future__ import annotations

import enum
import hashlib
import hmac
import secrets
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

PROTOCOL_VERSION = 1
SESSION_ID_SIZE = 16
HMAC_SIZE = 32
KEY_SIZE = 32
HEADER_SIZE = 1 + SESSION_ID_SIZE + 8 + 4  # 29 bytes

# Default cap ~25 MiB for JPEG payload (configurable at call sites via MAX_JPEG_SIZE)
MAX_JPEG_SIZE = 25 * 1024 * 1024
MIN_PACKAGE_SIZE = HEADER_SIZE + HMAC_SIZE  # zero-length JPEG still has header + tag


class VerifyReason(enum.Enum):
    ACCEPT = "ACCEPT"
    MALFORMED_PACKAGE = "MALFORMED_PACKAGE"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    HMAC_MISMATCH = "HMAC_MISMATCH"
    WRONG_SESSION = "WRONG_SESSION"
    REPLAY_OR_DUPLICATE = "REPLAY_OR_DUPLICATE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    INVALID_KEY = "INVALID_KEY"
    FRAME_TOO_LARGE = "FRAME_TOO_LARGE"


@dataclass(frozen=True)
class ParsedFramePackage:
    version: int
    session_id: bytes
    frame_counter: int
    jpeg_bytes: bytes
    hmac_tag: bytes

    @property
    def header(self) -> bytes:
        return serialize_header(
            self.version,
            self.session_id,
            self.frame_counter,
            len(self.jpeg_bytes),
        )


@dataclass(frozen=True)
class VerifyResult:
    reason: VerifyReason
    frame_counter: int | None = None
    session_id: bytes | None = None


def load_key(path: Path | str) -> bytes:
    """Load a raw 32-byte HMAC key from disk."""
    key_path = Path(path)
    if not key_path.is_file():
        raise ValueError(f"HMAC key file not found: {key_path}")
    key = key_path.read_bytes()
    if len(key) != KEY_SIZE:
        raise ValueError(
            f"HMAC key must be exactly {KEY_SIZE} bytes, got {len(key)}"
        )
    return key


def generate_session_id() -> bytes:
    """Return a new 128-bit session identifier."""
    return secrets.token_bytes(SESSION_ID_SIZE)


def _validate_session_id(session_id: bytes) -> None:
    if len(session_id) != SESSION_ID_SIZE:
        raise ValueError(
            f"session_id must be {SESSION_ID_SIZE} bytes, got {len(session_id)}"
        )


def _validate_frame_counter(frame_counter: int) -> None:
    if frame_counter < 0 or frame_counter > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("frame_counter must fit in unsigned 64-bit range")


def _validate_jpeg_length(jpeg_length: int, max_jpeg_size: int = MAX_JPEG_SIZE) -> None:
    if jpeg_length < 0 or jpeg_length > max_jpeg_size:
        raise ValueError(
            f"jpeg_length must be between 0 and {max_jpeg_size}, got {jpeg_length}"
        )


def serialize_header(
    version: int,
    session_id: bytes,
    frame_counter: int,
    jpeg_length: int,
    *,
    max_jpeg_size: int = MAX_JPEG_SIZE,
) -> bytes:
    """Deterministic big-endian header: version || session_id || counter || jpeg_length."""
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported version for serialization: {version}")
    _validate_session_id(session_id)
    _validate_frame_counter(frame_counter)
    _validate_jpeg_length(jpeg_length, max_jpeg_size)
    return (
        struct.pack(">B", version)
        + session_id
        + struct.pack(">QI", frame_counter, jpeg_length)
    )


def calculate_hmac(secret_key: bytes, header: bytes, jpeg_bytes: bytes) -> bytes:
    """HMAC-SHA256 over header || jpeg_bytes (tag not included)."""
    if len(secret_key) != KEY_SIZE:
        raise ValueError(f"secret_key must be {KEY_SIZE} bytes")
    data = header + jpeg_bytes
    return hmac.new(secret_key, data, hashlib.sha256).digest()


def create_frame_package(
    jpeg_bytes: bytes,
    session_id: bytes,
    frame_counter: int,
    secret_key: bytes,
    *,
    max_jpeg_size: int = MAX_JPEG_SIZE,
) -> bytes:
    """Build header || jpeg || HMAC tag."""
    if len(secret_key) != KEY_SIZE:
        raise ValueError(f"secret_key must be {KEY_SIZE} bytes")
    jpeg_length = len(jpeg_bytes)
    _validate_jpeg_length(jpeg_length, max_jpeg_size)

    header = serialize_header(
        PROTOCOL_VERSION,
        session_id,
        frame_counter,
        jpeg_length,
        max_jpeg_size=max_jpeg_size,
    )
    tag = calculate_hmac(secret_key, header, jpeg_bytes)
    return header + jpeg_bytes + tag


def parse_frame_package(
    package: bytes,
    *,
    max_jpeg_size: int = MAX_JPEG_SIZE,
) -> Tuple[ParsedFramePackage | None, VerifyReason | None]:
    """
    Parse and structurally validate a frame package.
    Returns (parsed, None) on success, or (None, reason) on failure.
    Does not verify HMAC or session/counter state.
    """
    if len(package) < MIN_PACKAGE_SIZE:
        return None, VerifyReason.MALFORMED_PACKAGE

    version = package[0]
    if version != PROTOCOL_VERSION:
        return None, VerifyReason.UNSUPPORTED_VERSION

    session_id = package[1 : 1 + SESSION_ID_SIZE]
    frame_counter, jpeg_length = struct.unpack(
        ">QI", package[1 + SESSION_ID_SIZE : HEADER_SIZE]
    )

    if jpeg_length > max_jpeg_size:
        return None, VerifyReason.FRAME_TOO_LARGE

    expected_len = HEADER_SIZE + jpeg_length + HMAC_SIZE
    if len(package) < expected_len:
        return None, VerifyReason.MALFORMED_PACKAGE
    if len(package) > expected_len:
        return None, VerifyReason.MALFORMED_PACKAGE

    jpeg_start = HEADER_SIZE
    jpeg_end = jpeg_start + jpeg_length
    jpeg_bytes = package[jpeg_start:jpeg_end]
    hmac_tag = package[jpeg_end : jpeg_end + HMAC_SIZE]

    if len(jpeg_bytes) != jpeg_length:
        return None, VerifyReason.MALFORMED_PACKAGE

    parsed = ParsedFramePackage(
        version=version,
        session_id=session_id,
        frame_counter=frame_counter,
        jpeg_bytes=jpeg_bytes,
        hmac_tag=hmac_tag,
    )
    return parsed, None


class FrameVerifier:
    """Stateful verifier: HMAC + session binding + monotonic frame counter."""

    def __init__(
        self,
        secret_key: bytes,
        expected_session_id: bytes,
        *,
        max_jpeg_size: int = MAX_JPEG_SIZE,
    ) -> None:
        if len(secret_key) != KEY_SIZE:
            raise ValueError(f"secret_key must be {KEY_SIZE} bytes")
        _validate_session_id(expected_session_id)
        self._secret_key = secret_key
        self._expected_session_id = expected_session_id
        self._max_jpeg_size = max_jpeg_size
        self._next_expected_counter = 0

    @property
    def next_expected_counter(self) -> int:
        return self._next_expected_counter

    def verify(self, package: bytes) -> VerifyResult:
        """Verify package; update counter only on ACCEPT."""
        parsed, parse_reason = parse_frame_package(
            package, max_jpeg_size=self._max_jpeg_size
        )
        if parse_reason is not None:
            return VerifyResult(reason=parse_reason)

        assert parsed is not None
        header = serialize_header(
            parsed.version,
            parsed.session_id,
            parsed.frame_counter,
            len(parsed.jpeg_bytes),
            max_jpeg_size=self._max_jpeg_size,
        )
        expected_tag = calculate_hmac(self._secret_key, header, parsed.jpeg_bytes)
        if not hmac.compare_digest(expected_tag, parsed.hmac_tag):
            return VerifyResult(reason=VerifyReason.HMAC_MISMATCH)

        if parsed.session_id != self._expected_session_id:
            return VerifyResult(
                reason=VerifyReason.WRONG_SESSION,
                frame_counter=parsed.frame_counter,
                session_id=parsed.session_id,
            )

        counter = parsed.frame_counter
        expected = self._next_expected_counter
        if counter < expected:
            return VerifyResult(
                reason=VerifyReason.REPLAY_OR_DUPLICATE,
                frame_counter=counter,
                session_id=parsed.session_id,
            )
        if counter > expected:
            return VerifyResult(
                reason=VerifyReason.OUT_OF_ORDER,
                frame_counter=counter,
                session_id=parsed.session_id,
            )

        self._next_expected_counter = expected + 1
        return VerifyResult(
            reason=VerifyReason.ACCEPT,
            frame_counter=counter,
            session_id=parsed.session_id,
        )

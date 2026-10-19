"""Local TCP framing: Webots controller → VIGIL bridge (BGR, not HMAC)."""

from __future__ import annotations

import socket
import struct

import numpy as np

MAGIC = b"VGBR"
_HEADER = struct.Struct(">4sII")


def pack_bgr_frame(frame_bgr: np.ndarray) -> bytes:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("Expected HxWx3 BGR uint8 frame")
    h, w = frame_bgr.shape[:2]
    raw = frame_bgr.astype(np.uint8, copy=False).tobytes()
    return _HEADER.pack(MAGIC, w, h) + raw


def recv_bgr_frame(sock: socket.socket) -> np.ndarray | None:
    header = _recv_exact(sock, _HEADER.size)
    if header is None:
        return None
    magic, w, h = _HEADER.unpack(header)
    if magic != MAGIC or w <= 0 or h <= 0 or w > 4096 or h > 4096:
        return None
    nbytes = w * h * 3
    payload = _recv_exact(sock, nbytes)
    if payload is None or len(payload) != nbytes:
        return None
    return np.frombuffer(payload, dtype=np.uint8).reshape((h, w, 3)).copy()


def send_bgr_frame(sock: socket.socket, frame_bgr: np.ndarray) -> None:
    sock.sendall(pack_bgr_frame(frame_bgr))


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        try:
            chunk = sock.recv(remaining)
        except OSError:
            return None
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

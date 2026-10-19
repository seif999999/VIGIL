"""Automated: frame protocol + HMAC stream hook (no Webots GUI required)."""

from __future__ import annotations

import socket
import struct
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
HMAC_DIR = BASE_DIR.parent / "hmac"
sys.path.insert(0, str(HMAC_DIR))
sys.path.insert(0, str(BASE_DIR))

from frame_protocol import pack_bgr_frame, recv_bgr_frame  # noqa: E402
from protocol import (  # noqa: E402
    VerifyReason,
    create_frame_package,
    generate_session_id,
    load_key,
    FrameVerifier,
    HEADER_SIZE,
    HMAC_SIZE,
)


def test_frame_protocol_roundtrip() -> bool:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[30:90, 40:120] = (40, 180, 220)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    received: list[np.ndarray] = []

    def client() -> None:
        c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        c.connect(("127.0.0.1", port))
        c.sendall(pack_bgr_frame(frame))
        c.close()

    threading.Thread(target=client, daemon=True).start()
    conn, _ = srv.accept()
    got = recv_bgr_frame(conn)
    conn.close()
    srv.close()
    return got is not None and got.shape == frame.shape and np.array_equal(got, frame)


def test_hmac_from_bgr_like_webots() -> bool:
    image_path = HMAC_DIR / "test_images" / "original_1.jpg"
    key_path = HMAC_DIR / "secrets" / "hmac.key"
    if not image_path.is_file() or not key_path.is_file():
        print("[SKIP] hmac test image or key missing")
        return True
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return False
    jpeg = cv2.imencode(".jpg", bgr)[1].tobytes()
    key = load_key(key_path)
    session = generate_session_id()
    pkg = create_frame_package(jpeg, session, 0, key)
    result = FrameVerifier(key, session).verify(pkg)
    if result.reason != VerifyReason.ACCEPT:
        return False
    verified_jpeg = pkg[HEADER_SIZE : len(pkg) - HMAC_SIZE]
    return verified_jpeg == jpeg


def main() -> None:
    ok_proto = test_frame_protocol_roundtrip()
    ok_hmac = test_hmac_from_bgr_like_webots()
    print("--- VIGIL Webots bridge (automated, no GUI) ---")
    print(f"frame_protocol_roundtrip: {'PASS' if ok_proto else 'FAIL'}")
    print(f"hmac_from_bgr_jpeg: {'PASS' if ok_hmac else 'FAIL'}")
    ok = ok_proto and ok_hmac
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

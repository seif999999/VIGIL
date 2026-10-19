"""Verify SCRFD runs only after HMAC ACCEPT (synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

HMAC_DIR = Path(__file__).resolve().parent.parent / "hmac"
sys.path.insert(0, str(HMAC_DIR))

from protocol import (  # noqa: E402
    VerifyReason,
    create_frame_package,
    generate_session_id,
    load_key,
    FrameVerifier,
    HEADER_SIZE,
    HMAC_SIZE,
)

from detector import FaceDetector  # noqa: E402


class CountingDetector(FaceDetector):
    calls = 0

    def detect(self, frame_bgr: np.ndarray):
        CountingDetector.calls += 1
        return super().detect(frame_bgr)


def simulate_verifier_path(jpeg: bytes, packages: list[bytes], key: bytes, session: bytes) -> tuple[int, int]:
    """Returns (verified_count, rejected_count). Invokes detector only on ACCEPT."""
    CountingDetector.calls = 0
    detector = CountingDetector()
    verifier = FrameVerifier(key, session)
    verified = 0
    rejected = 0

    for package in packages:
        result = verifier.verify(package)
        if result.reason != VerifyReason.ACCEPT:
            rejected += 1
            continue
        verified += 1
        verified_jpeg = package[HEADER_SIZE : len(package) - HMAC_SIZE]
        buf = np.frombuffer(verified_jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is not None:
            detector.detect(bgr)

    return verified, rejected


def main() -> None:
    image_path = HMAC_DIR / "test_images" / "original_1.jpg"
    key_path = HMAC_DIR / "secrets" / "hmac.key"
    if not image_path.is_file() or not key_path.is_file():
        print("[SKIP] Need original_1.jpg and hmac.key")
        sys.exit(0)

    jpeg = image_path.read_bytes()
    key = load_key(key_path)
    session = generate_session_id()

    valid = [create_frame_package(jpeg, session, i, key) for i in range(20)]
    tampered = bytearray(valid[19])
    tampered[HEADER_SIZE] ^= 0x01
    packages = valid[:19] + [bytes(tampered)]

    verified, rejected = simulate_verifier_path(jpeg, packages, key, session)

    print("--- VIGIL Security Gate (synthetic) ---")
    print(f"Verified frames: {verified}")
    print(f"Rejected frames: {rejected}")
    print(f"Detector calls: {CountingDetector.calls}")
    print(f"Reject reason expected: HMAC_MISMATCH on last frame")

    ok = verified == 19 and rejected == 1 and CountingDetector.calls == 19
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

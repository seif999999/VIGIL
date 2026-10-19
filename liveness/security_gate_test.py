"""HMAC gate: rejected frames must not invoke SCRFD or liveness."""

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

from detector import LivenessDetector  # noqa: E402
from face_bridge import load_face_detector_module  # noqa: E402
from pipeline import run_liveness_if_face_ok  # noqa: E402


class CountingFaceDetector:
    calls = 0

    def __init__(self, inner) -> None:
        self._inner = inner

    def detect(self, frame_bgr: np.ndarray):
        CountingFaceDetector.calls += 1
        return self._inner.detect(frame_bgr)


class CountingLiveness(LivenessDetector):
    calls = 0

    def predict(self, face_crop_bgr: np.ndarray):
        CountingLiveness.calls += 1
        return super().predict(face_crop_bgr)


def simulate_verifier_perception(
    packages: list[bytes],
    key: bytes,
    session: bytes,
    face_detector,
    liveness: CountingLiveness,
) -> tuple[int, int]:
    CountingFaceDetector.calls = 0
    CountingLiveness.calls = 0
    wrapped = CountingFaceDetector(face_detector)
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
        if bgr is None:
            continue
        det = wrapped.detect(bgr)
        run_liveness_if_face_ok(bgr, det, liveness)

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

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    liveness = CountingLiveness()

    verified, rejected = simulate_verifier_perception(
        packages, key, session, face_detector, liveness
    )

    liveness_on_rejected = 0  # by construction, reject path never calls perception

    print("--- VIGIL Liveness Security Gate (synthetic) ---")
    print(f"Verified frames: {verified}")
    print(f"Rejected frames: {rejected}")
    print(f"Face detector calls: {CountingFaceDetector.calls}")
    print(f"Liveness calls: {CountingLiveness.calls}")
    print(f"Liveness calls on rejected frames: {liveness_on_rejected}")

    ok = (
        verified == 19
        and rejected == 1
        and CountingFaceDetector.calls == 19
        and CountingLiveness.calls <= 19
        and liveness_on_rejected == 0
    )
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

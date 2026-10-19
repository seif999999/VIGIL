"""Integrated HMAC → SCRFD → liveness → provisional policy (automated, no camera)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
LIVENESS_DIR = BASE_DIR.parent / "liveness"
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(LIVENESS_DIR))

from protocol import (  # noqa: E402
    VerifyReason,
    create_frame_package,
    generate_session_id,
    load_key,
    FrameVerifier,
    HEADER_SIZE,
    HMAC_SIZE,
)
from decision_policy import (  # noqa: E402
    ProvisionalDecisionPolicy,
    ProvisionalDemoOutcome,
    resolve_demo_outcome,
)
from pipeline import run_liveness_if_face_ok  # noqa: E402


class CountingFaceDetector:
    calls = 0

    def __init__(self, inner) -> None:
        self._inner = inner

    def detect(self, frame_bgr: np.ndarray):
        CountingFaceDetector.calls += 1
        return self._inner.detect(frame_bgr)


class CountingLiveness:
    calls = 0

    def predict_from_frame_bbox(self, frame_bgr, bbox):
        CountingLiveness.calls += 1
        return self._inner.predict_from_frame_bbox(frame_bgr, bbox)

    def __init__(self, inner) -> None:
        self._inner = inner


def process_packages(
    packages: list[bytes],
    key: bytes,
    session: bytes,
    face_detector,
    liveness,
    policy: ProvisionalDecisionPolicy,
) -> tuple[int, int, dict[str, int]]:
    """Verifier-side path: HMAC verify then perception only on ACCEPT."""
    CountingFaceDetector.calls = 0
    CountingLiveness.calls = 0
    wrapped_face = CountingFaceDetector(face_detector)
    wrapped_live = CountingLiveness(liveness)
    verifier = FrameVerifier(key, session)
    verified = 0
    rejected = 0
    demo_counts = {o.value: 0 for o in ProvisionalDemoOutcome}

    for package in packages:
        result = verifier.verify(package)
        if result.reason != VerifyReason.ACCEPT:
            rejected += 1
            continue
        verified += 1
        verified_jpeg = package[HEADER_SIZE : len(package) - HMAC_SIZE]
        bgr = cv2.imdecode(
            np.frombuffer(verified_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if bgr is None:
            continue
        det = wrapped_face.detect(bgr)
        live = run_liveness_if_face_ok(bgr, det, wrapped_live)
        demo = resolve_demo_outcome(det.decision.value, live, policy)
        demo_counts[demo.value] += 1

    return verified, rejected, demo_counts


def test_tampered_hmac_no_perception(
    jpeg: bytes, key: bytes, session: bytes, face_detector, liveness
) -> bool:
    valid = [create_frame_package(jpeg, session, i, key) for i in range(5)]
    tampered = bytearray(valid[-1])
    tampered[HEADER_SIZE] ^= 0x01
    packages = valid[:-1] + [bytes(tampered)]
    policy = ProvisionalDecisionPolicy()
    verified, rejected, _ = process_packages(
        packages, key, session, face_detector, liveness, policy
    )
    return (
        verified == 4
        and rejected == 1
        and CountingFaceDetector.calls == 4
        and CountingLiveness.calls <= 4
    )


def test_replay_no_extra_perception(
    jpeg: bytes, key: bytes, session: bytes, face_detector, liveness
) -> bool:
    pkg = create_frame_package(jpeg, session, 0, key)
    policy = ProvisionalDecisionPolicy()
    verified, rejected, _ = process_packages(
        [pkg, pkg], key, session, face_detector, liveness, policy
    )
    return verified == 1 and rejected == 1 and CountingFaceDetector.calls == 1


def test_no_face_skips_liveness(face_mod, liveness) -> bool:
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    policy = ProvisionalDecisionPolicy()
    det = face_mod.FaceDetector().detect(blank)
    if det.decision.value != "NO_FACE":
        return False
    CountingLiveness.calls = 0
    stub = CountingLiveness(liveness)
    live = run_liveness_if_face_ok(blank, det, stub)
    demo = resolve_demo_outcome(det.decision.value, live, policy)
    return live is None and CountingLiveness.calls == 0 and demo == ProvisionalDemoOutcome.NO_FACE


def test_multiple_faces_skips_liveness(face_mod, liveness) -> bool:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    faces = [
        SimpleNamespace(bbox=(40.0, 40.0, 160.0, 160.0)),
        SimpleNamespace(bbox=(120.0, 40.0, 240.0, 160.0)),
    ]
    det = SimpleNamespace(
        faces=tuple(faces),
        face_count=2,
        decision=face_mod.face_count_decision(2),
    )
    CountingLiveness.calls = 0
    stub = CountingLiveness(liveness)
    live = run_liveness_if_face_ok(frame, det, stub)
    policy = ProvisionalDecisionPolicy()
    demo = resolve_demo_outcome(det.decision.value, live, policy)
    return (
        live is None
        and CountingLiveness.calls == 0
        and demo == ProvisionalDemoOutcome.MULTIPLE_FACES
    )


def test_policy_on_verified_frame(jpeg: bytes, key: bytes, session: bytes, face_detector, liveness) -> bool:
    pkg = create_frame_package(jpeg, session, 0, key)
    policy = ProvisionalDecisionPolicy(0.85, 0.85)
    _, _, demo_counts = process_packages(
        [pkg], key, session, face_detector, liveness, policy
    )
    total_demo = sum(demo_counts.values())
    if total_demo != 1:
        return False
    allowed = {
        ProvisionalDemoOutcome.REAL.value,
        ProvisionalDemoOutcome.SPOOF.value,
        ProvisionalDemoOutcome.UNCERTAIN_REVIEW.value,
        ProvisionalDemoOutcome.NO_FACE.value,
        ProvisionalDemoOutcome.MULTIPLE_FACES.value,
    }
    return all(k in allowed for k in demo_counts)


def main() -> None:
    image_path = BASE_DIR / "test_images" / "original_1.jpg"
    key_path = BASE_DIR / "secrets" / "hmac.key"
    if not image_path.is_file() or not key_path.is_file():
        print("[SKIP] Need test_images/original_1.jpg and secrets/hmac.key")
        sys.exit(0)

    jpeg = image_path.read_bytes()
    key = load_key(key_path)
    session = generate_session_id()

    from detector import LivenessDetector  # noqa: WPS433
    from face_bridge import load_face_detector_module  # noqa: WPS433

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    liveness = LivenessDetector()

    results: list[tuple[str, bool]] = []
    results.append(
        (
            "tampered_hmac_fail_closed",
            test_tampered_hmac_no_perception(jpeg, key, session, face_detector, liveness),
        )
    )
    session2 = generate_session_id()
    results.append(
        (
            "replay_rejected_before_second_detect",
            test_replay_no_extra_perception(jpeg, key, session2, face_detector, liveness),
        )
    )
    results.append(("no_face_skips_liveness", test_no_face_skips_liveness(face_mod, liveness)))
    results.append(
        (
            "multiple_faces_skips_liveness",
            test_multiple_faces_skips_liveness(face_mod, liveness),
        )
    )
    session3 = generate_session_id()
    results.append(
        (
            "provisional_policy_on_accepted_frame",
            test_policy_on_verified_frame(jpeg, key, session3, face_detector, liveness),
        )
    )

    print("--- VIGIL Integrated Visual Baseline (automated) ---")
    all_ok = True
    for name, ok in results:
        print(f"{name}: {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
    print(f"Result: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""Liveness runs only when face-count decision is FACE_OK (exactly one face)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from detector import LivenessDetector
from face_bridge import load_face_detector_module
from pipeline import run_liveness_if_face_ok


class CountingLiveness(LivenessDetector):
    calls = 0

    def predict(self, face_crop_bgr: np.ndarray):
        CountingLiveness.calls += 1
        return super().predict(face_crop_bgr)


def _fake_detection(face_mod, count: int):
    faces = []
    for i in range(count):
        x = 40 + i * 80
        faces.append(
            SimpleNamespace(
                x1=x,
                y1=40,
                x2=x + 120,
                y2=160,
                bbox=(float(x), 40.0, float(x + 120), 160.0),
            )
        )
    decision = face_mod.face_count_decision(count)
    return SimpleNamespace(faces=tuple(faces), face_count=count, decision=decision)


def main() -> None:
    face_mod = load_face_detector_module()
    liveness = CountingLiveness()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[40:160, 40:160] = 128

    CountingLiveness.calls = 0
    run_liveness_if_face_ok(frame, _fake_detection(face_mod, 0), liveness)
    after_no_face = CountingLiveness.calls

    CountingLiveness.calls = 0
    run_liveness_if_face_ok(frame, _fake_detection(face_mod, 2), liveness)
    after_multi = CountingLiveness.calls

    CountingLiveness.calls = 0
    run_liveness_if_face_ok(frame, _fake_detection(face_mod, 1), liveness)
    after_one = CountingLiveness.calls

    print("--- VIGIL Face-Count Gate (synthetic) ---")
    print(f"NO_FACE liveness calls: {after_no_face} (expected 0)")
    print(f"MULTIPLE_FACES liveness calls: {after_multi} (expected 0)")
    print(f"FACE_OK liveness calls: {after_one} (expected 1)")

    ok = after_no_face == 0 and after_multi == 0 and after_one == 1
    print(f"Result: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

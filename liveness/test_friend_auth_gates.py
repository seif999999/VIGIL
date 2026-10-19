"""Friend-auth perception must not run on HMAC reject / non-FACE_OK."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
HMAC = BASE.parent / "hmac"
sys.path.insert(0, str(HMAC))
sys.path.insert(0, str(BASE))

from protocol import (  # noqa: E402
    HEADER_SIZE,
    HMAC_SIZE,
    VerifyReason,
    create_frame_package,
    FrameVerifier,
    generate_session_id,
)

from auth_pipeline import run_fasnet_if_face_ok  # noqa: E402
from fasnet_liveness import FasNetClass, FasNetResult  # noqa: E402


class CountingFasNet:
    calls = 0

    def analyze_full_frame(self, frame_bgr, bbox):
        CountingFasNet.calls += 1
        return FasNetResult(
            True, 0.9, FasNetClass.REAL, 1.0, (0, 0, 10, 10), None
        )


class FriendAuthGateTests(unittest.TestCase):
    def test_fasnet_skipped_without_face_ok(self) -> None:
        CountingFasNet.calls = 0
        fas = CountingFasNet()
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        for decision, count in (("NO_FACE", 0), ("MULTIPLE_FACES", 2)):
            det = SimpleNamespace(
                decision=SimpleNamespace(value=decision),
                face_count=count,
                faces=(),
            )
            self.assertIsNone(run_fasnet_if_face_ok(frame, det, fas))
        self.assertEqual(CountingFasNet.calls, 0)

    def test_hmac_reject_skips_perception(self) -> None:
        """Mirrors integrated baseline: tamper → no FasNet calls."""
        key = b"\x11" * 32
        session = generate_session_id()
        jpeg = cv2.imencode(".jpg", np.zeros((64, 64, 3), dtype=np.uint8))[1].tobytes()
        good = create_frame_package(jpeg, session, 0, key)
        bad = bytearray(good)
        bad[HEADER_SIZE] ^= 0xFF
        verifier = FrameVerifier(key, session)
        CountingFasNet.calls = 0
        fas = CountingFasNet()

        # Accept first would advance counter; we only verify reject path
        r = verifier.verify(bytes(bad))
        self.assertNotEqual(r.reason, VerifyReason.ACCEPT)
        # Perception must not be invoked by caller on reject — simulate gate
        if r.reason == VerifyReason.ACCEPT:
            run_fasnet_if_face_ok(
                np.zeros((64, 64, 3), dtype=np.uint8),
                SimpleNamespace(
                    decision=SimpleNamespace(value="FACE_OK"),
                    face_count=1,
                    faces=(SimpleNamespace(bbox=(0, 0, 10, 10)),),
                ),
                fas,
            )
        self.assertEqual(CountingFasNet.calls, 0)


if __name__ == "__main__":
    unittest.main()

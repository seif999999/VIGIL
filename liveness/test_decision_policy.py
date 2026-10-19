"""Unit tests for provisional demo liveness outcomes (no ONNX)."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace

from decision_policy import (
    ProvisionalDecisionPolicy,
    ProvisionalDemoOutcome,
    classify_provisional_scores,
    resolve_demo_outcome,
)
from detector import LivenessResult, PredictedClass
from face_bridge import load_face_detector_module
from pipeline import run_liveness_if_face_ok


def _live(real: float, spoof: float) -> LivenessResult:
    pred = PredictedClass.REAL if real > spoof else PredictedClass.SPOOF
    return LivenessResult(real, spoof, pred, 1.0)


class TestProvisionalScorePolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ProvisionalDecisionPolicy(0.85, 0.85)

    def test_confident_real(self) -> None:
        out = classify_provisional_scores(0.95, 0.05, self.policy)
        self.assertEqual(out, ProvisionalDemoOutcome.REAL)

    def test_confident_spoof(self) -> None:
        out = classify_provisional_scores(0.04, 0.96, self.policy)
        self.assertEqual(out, ProvisionalDemoOutcome.SPOOF)

    def test_uncertain_review(self) -> None:
        out = classify_provisional_scores(0.52, 0.48, self.policy)
        self.assertEqual(out, ProvisionalDemoOutcome.UNCERTAIN_REVIEW)


class TestFaceGateOutcomes(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ProvisionalDecisionPolicy()

    def test_no_face(self) -> None:
        out = resolve_demo_outcome("NO_FACE", None, self.policy)
        self.assertEqual(out, ProvisionalDemoOutcome.NO_FACE)

    def test_multiple_faces(self) -> None:
        out = resolve_demo_outcome("MULTIPLE_FACES", None, self.policy)
        self.assertEqual(out, ProvisionalDemoOutcome.MULTIPLE_FACES)

    def test_face_ok_with_scores(self) -> None:
        out = resolve_demo_outcome(
            "FACE_OK", _live(0.95, 0.05), self.policy
        )
        self.assertEqual(out, ProvisionalDemoOutcome.REAL)


class TestPipelineFaceGate(unittest.TestCase):
    def test_liveness_not_run_without_face_ok(self) -> None:
        face_mod = load_face_detector_module()

        class StubLiveness:
            calls = 0

            def predict_from_frame_bbox(self, frame_bgr, bbox):
                StubLiveness.calls += 1
                return _live(0.5, 0.5)

        stub = StubLiveness()
        frame = __import__("numpy").zeros((240, 320, 3), dtype=__import__("numpy").uint8)

        def fake(count: int):
            faces = []
            for i in range(count):
                x = 40 + i * 80
                faces.append(SimpleNamespace(bbox=(float(x), 40.0, float(x + 120), 160.0)))
            return SimpleNamespace(
                faces=tuple(faces),
                face_count=count,
                decision=face_mod.face_count_decision(count),
            )

        StubLiveness.calls = 0
        self.assertIsNone(run_liveness_if_face_ok(frame, fake(0), stub))
        self.assertEqual(StubLiveness.calls, 0)

        StubLiveness.calls = 0
        self.assertIsNone(run_liveness_if_face_ok(frame, fake(2), stub))
        self.assertEqual(StubLiveness.calls, 0)


def main() -> None:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()

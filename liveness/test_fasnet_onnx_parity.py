"""Parity: ONNX FasNet vs TF FasNet on local enrollment photos (no uploads)."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from facenet_identity import DEFAULT_ENROLLMENT_PATH, load_enrollments
from fasnet_liveness import FasNetLiveness


def _seif_images(limit: int = 4) -> list[np.ndarray]:
    db = load_enrollments(DEFAULT_ENROLLMENT_PATH)
    paths = list((db.get("identities") or {}).get("Seif", {}).get("source_images") or [])
    out: list[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            out.append(img)
        if len(out) >= limit:
            break
    return out


@unittest.skipUnless(DEFAULT_ENROLLMENT_PATH.is_file(), "enrollments.json missing")
class FasNetOnnxParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed")
        from face_bridge import load_face_detector_module

        cls.images = _seif_images(4)
        if len(cls.images) < 2:
            raise unittest.SkipTest("need Seif enrollment photos")
        face_mod = load_face_detector_module()
        cls.detector = face_mod.FaceDetector()
        cls.onnx = FasNetLiveness(backend="onnx")
        cls.onnx.warm_up()
        cls.tf = FasNetLiveness(backend="tf")
        cls.tf.warm_up()

    def test_backend_labels(self) -> None:
        self.assertEqual(self.onnx.backend, "onnx")
        self.assertEqual(self.tf.backend, "tf")

    def test_class_and_score_parity(self) -> None:
        max_score_delta = 0.0
        max_avg_delta = 0.0
        for img in self.images:
            det = self.detector.detect(img)
            if det.face_count != 1:
                continue
            bbox = det.faces[0].bbox
            a = self.onnx.analyze_full_frame(img, bbox)
            b = self.tf.analyze_full_frame(img, bbox)
            self.assertIsNone(a.error, a.error)
            self.assertIsNone(b.error, b.error)
            self.assertEqual(a.is_real, b.is_real)
            self.assertEqual(a.predicted_class, b.predicted_class)
            assert a.antispoof_score is not None and b.antispoof_score is not None
            score_delta = abs(a.antispoof_score - b.antispoof_score)
            max_score_delta = max(max_score_delta, score_delta)
            assert a.class_scores_avg is not None and b.class_scores_avg is not None
            avg_delta = max(
                abs(x - y) for x, y in zip(a.class_scores_avg, b.class_scores_avg)
            )
            max_avg_delta = max(max_avg_delta, avg_delta)
            # Tight tolerance: observed ~1e-8 on this laptop; allow float32 noise.
            self.assertLess(score_delta, 1e-4)
            self.assertLess(avg_delta, 1e-4)
        self.assertGreater(max_score_delta, -1.0)  # ran at least one comparison


if __name__ == "__main__":
    unittest.main()

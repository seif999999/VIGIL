"""Parity: same-weight ONNX FaceNet vs TF on local Seif photos (no uploads)."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from facenet_backends import DEFAULT_ONNX_PATH, OnnxFaceNetForward, TfFaceNetForward, preprocess_facenet_input
from facenet_identity import (
    DEFAULT_ENROLLMENT_PATH,
    FaceNetIdentity,
    build_enrollment_index,
    load_enrollments,
    match_identity,
)


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
@unittest.skipUnless(DEFAULT_ONNX_PATH.is_file(), "FaceNet ONNX export missing")
class FaceNetOnnxParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed")
        cls.images = _seif_images(4)
        if len(cls.images) < 2:
            raise unittest.SkipTest("need Seif enrollment photos")
        cls.db = load_enrollments()
        cls.index = build_enrollment_index(cls.db)
        cls.tf = FaceNetIdentity(backend="tf")
        cls.onnx = FaceNetIdentity(backend="onnx")
        cls.tf.warm_up()
        cls.onnx.warm_up()

    def test_backend_labels(self) -> None:
        self.assertEqual(self.tf.backend, "tf")
        self.assertEqual(self.onnx.backend, "onnx")

    def test_same_tensor_forward_parity(self) -> None:
        batch, _ = preprocess_facenet_input(self.images[0])
        tf_fwd = TfFaceNetForward()
        onnx_fwd = OnnxFaceNetForward()
        et, _ = tf_fwd.forward(batch)
        eo, _ = onnx_fwd.forward(batch)
        max_abs = float(
            np.max(np.abs(np.asarray(et, dtype=np.float64) - np.asarray(eo, dtype=np.float64)))
        )
        self.assertLess(max_abs, 1e-4)

    def test_embedding_and_decision_parity(self) -> None:
        max_abs = 0.0
        max_dist_delta = 0.0
        compared = 0
        for img in self.images:
            et, _, e1 = self.tf.embed_bgr(img)
            eo, _, e2 = self.onnx.embed_bgr(img)
            self.assertIsNone(e1, e1)
            self.assertIsNone(e2, e2)
            assert et is not None and eo is not None
            delta = float(
                np.max(np.abs(np.asarray(et, dtype=np.float64) - np.asarray(eo, dtype=np.float64)))
            )
            max_abs = max(max_abs, delta)
            mt = match_identity(et, self.db, enrollment_index=self.index)
            mo = match_identity(eo, self.db, enrollment_index=self.index)
            self.assertEqual(mt.recognized, mo.recognized)
            self.assertEqual(mt.identity, mo.identity)
            if mt.best_distance is not None and mo.best_distance is not None:
                max_dist_delta = max(max_dist_delta, abs(mt.best_distance - mo.best_distance))
            compared += 1
        self.assertGreaterEqual(compared, 2)
        self.assertLess(max_abs, 1e-4)
        self.assertLess(max_dist_delta, 1e-3)


if __name__ == "__main__":
    unittest.main()

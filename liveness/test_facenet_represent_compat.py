"""Regression: DeepFace.represent kwargs must match the installed API.

deepface==0.0.102 rejects silent=True (TypeError). FaceNetIdentity must filter
unsupported kwargs so identity inference runs instead of fail-closing on a
compatibility bug.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from facenet_identity import FaceNetIdentity, _represent_kwargs


def _fake_represent_no_silent(
    img_path,
    model_name="VGG-Face",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    expand_percentage=0,
    normalization="base",
    anti_spoofing=False,
    max_faces=None,
    l2_normalize=False,
    minmax_normalize=False,
    return_face=False,
):
    """Mirrors deepface==0.0.102 represent() — no silent kwarg."""
    return [{"embedding": [0.0] * 128, "facial_area": {}, "face_confidence": 0.0}]


def _fake_represent_with_silent(
    img_path,
    model_name="VGG-Face",
    enforce_detection=True,
    detector_backend="opencv",
    align=True,
    expand_percentage=0,
    normalization="base",
    anti_spoofing=False,
    max_faces=None,
    l2_normalize=False,
    minmax_normalize=False,
    return_face=False,
    silent=False,
):
    return [{"embedding": [1.0] * 128, "facial_area": {}, "face_confidence": 0.0}]


class RepresentKwargsCompatTests(unittest.TestCase):
    def test_filters_silent_for_0_0_102_signature(self) -> None:
        kwargs = _represent_kwargs(
            _fake_represent_no_silent,
            img_path=np.zeros((10, 10, 3), dtype=np.uint8),
            model_name="Facenet",
            enforce_detection=False,
            silent=True,
        )
        self.assertNotIn("silent", kwargs)
        self.assertEqual(kwargs["model_name"], "Facenet")
        self.assertFalse(kwargs["enforce_detection"])
        out = _fake_represent_no_silent(**kwargs)
        self.assertEqual(len(out[0]["embedding"]), 128)

    def test_keeps_silent_when_supported(self) -> None:
        kwargs = _represent_kwargs(
            _fake_represent_with_silent,
            img_path="x.jpg",
            model_name="Facenet",
            enforce_detection=True,
            silent=True,
        )
        self.assertIn("silent", kwargs)
        self.assertTrue(kwargs["silent"])

    def test_installed_deepface_represent_rejects_raw_silent(self) -> None:
        """Document installed API: silent is unsupported on deepface 0.0.102."""
        try:
            from deepface import DeepFace
        except ImportError:
            self.skipTest("deepface not installed")
        params = inspect.signature(DeepFace.represent).parameters
        if "silent" in params:
            self.skipTest("installed DeepFace already accepts silent")
        with self.assertRaises(TypeError):
            DeepFace.represent(
                img_path=np.zeros((32, 32, 3), dtype=np.uint8),
                model_name="Facenet",
                enforce_detection=False,
                silent=True,
            )

    def test_embed_bgr_does_not_surface_silent_typeerror(self) -> None:
        """embed_bgr no longer calls DeepFace.represent(silent=...); must not TypeError."""
        from facenet_backends import FaceNetStageTiming, DEFAULT_ONNX_PATH

        backend = "onnx" if DEFAULT_ONNX_PATH.is_file() else "tf"
        identity = FaceNetIdentity(backend=backend)
        identity._warmed = True
        fake_pre = FaceNetStageTiming(0, 0, 0, 0, 0, 0, "preprocess")
        fake_emb = [0.0] * 128

        def fake_forward(_batch):
            return fake_emb, 1.0

        if backend == "onnx":
            identity._onnx_fwd.forward = fake_forward  # type: ignore[method-assign]
        else:
            identity._tf_fwd.forward = fake_forward  # type: ignore[method-assign]

        with unittest.mock.patch(
            "facenet_backends.preprocess_facenet_input",
            return_value=(np.zeros((1, 160, 160, 3), dtype=np.float32), fake_pre),
        ):
            emb, _ms, err = identity.embed_bgr(
                np.zeros((80, 80, 3), dtype=np.uint8), enforce_detection=False
            )

        self.assertIsNone(err, msg=f"unexpected error: {err}")
        self.assertIsNotNone(emb)
        assert emb is not None
        self.assertEqual(len(emb), 128)
        self.assertNotIn("unexpected keyword", (err or "").lower())


class FailClosedTests(unittest.TestCase):
    def test_forward_exceptions_still_fail_closed(self) -> None:
        """Inference errors must return (None, ms, error), never raise."""
        from facenet_backends import FaceNetStageTiming

        identity = FaceNetIdentity(backend="tf")
        identity._warmed = True

        def boom(_batch):
            raise RuntimeError("inference_failed")

        identity._tf_fwd.forward = boom  # type: ignore[method-assign]
        fake_pre = FaceNetStageTiming(0, 0, 0, 0, 0, 0, "preprocess")
        with unittest.mock.patch(
            "facenet_backends.preprocess_facenet_input",
            return_value=(np.zeros((1, 160, 160, 3), dtype=np.float32), fake_pre),
        ):
            emb, _ms, err = identity.embed_bgr(
                np.zeros((40, 40, 3), dtype=np.uint8), enforce_detection=False
            )

        self.assertIsNone(emb)
        self.assertIsNotNone(err)
        self.assertIn("RuntimeError", err or "")


if __name__ == "__main__":
    unittest.main()

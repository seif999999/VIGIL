"""anti-spoof-mn3 liveness classifier (Open Model Zoo, perception only)."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from crop import crop_face_bgr

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "anti-spoof-mn3.onnx"
MODEL_NAME = "anti-spoof-mn3"
INPUT_SIZE = (128, 128)

# Open Model Zoo anti-spoof-mn3 — Original Model (ONNX) preprocessing.
# See models/public/anti-spoof-mn3/README.md in openvinotoolkit/open_model_zoo.
_MEAN_RGB = np.array([151.2405, 119.5950, 107.8395], dtype=np.float32)
_SCALE_RGB = np.array([63.0105, 56.4570, 55.0035], dtype=np.float32)


class PredictedClass(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"


@dataclass(frozen=True)
class LivenessResult:
    real_score: float
    spoof_score: float
    predicted_class: PredictedClass
    inference_ms: float


class LivenessDetector:
    """Binary face anti-spoofing on a face crop (128x128). No HMAC or SCRFD."""

    def __init__(self, model_path: Path | str | None = None) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"anti-spoof-mn3 ONNX not found at {path}. See liveness/models/MODEL_INFO.txt"
            )
        self._model_path = path
        self._session = ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != "actual_input_1":
            raise RuntimeError(f"Unexpected ONNX inputs: {inputs}")
        if len(outputs) != 1 or outputs[0].name != "output1":
            raise RuntimeError(f"Unexpected ONNX outputs: {outputs}")
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

    @property
    def model_path(self) -> Path:
        return self._model_path

    @staticmethod
    def preprocess_face_crop_bgr(crop_bgr: np.ndarray) -> np.ndarray:
        """NCHW float32 tensor for official OMZ ONNX (RGB, mean/scale)."""
        resized = cv2.resize(crop_bgr, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb - _MEAN_RGB) / _SCALE_RGB
        nchw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(nchw, axis=0)

    def predict(self, face_crop_bgr: np.ndarray) -> LivenessResult:
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            raise ValueError("Empty face crop")

        blob = self.preprocess_face_crop_bgr(face_crop_bgr)
        t0 = time.perf_counter_ns()
        (raw,) = self._session.run([self._output_name], {self._input_name: blob})
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000

        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
        if scores.shape[0] != 2:
            raise RuntimeError(f"Expected 2-class output, got shape {scores.shape}")

        real_score = float(scores[0])
        spoof_score = float(scores[1])
        if real_score + spoof_score > 1.01 or real_score + spoof_score < 0.99:
            exp = np.exp(scores - np.max(scores))
            prob = exp / np.sum(exp)
            real_score = float(prob[0])
            spoof_score = float(prob[1])

        predicted = (
            PredictedClass.REAL
            if real_score > spoof_score
            else PredictedClass.SPOOF
        )
        return LivenessResult(real_score, spoof_score, predicted, elapsed_ms)

    def predict_from_frame_bbox(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> LivenessResult | None:
        crop = crop_face_bgr(frame_bgr, bbox)
        if crop is None:
            return None
        return self.predict(crop)

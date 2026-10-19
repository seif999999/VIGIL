"""SCRFD-2.5G_KPS face detection for VIGIL (perception layer only)."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo.scrfd import SCRFD

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    BASE_DIR / "models" / "buffalo_m" / "buffalo_m" / "det_2.5g.onnx"
)
DETECTOR_NAME = "SCRFD-2.5G_KPS"
DEFAULT_INPUT_SIZE = (640, 640)
DEFAULT_DET_THRESH = 0.5


class FaceCountDecision(str, enum.Enum):
    NO_FACE = "NO_FACE"
    FACE_OK = "FACE_OK"
    MULTIPLE_FACES = "MULTIPLE_FACES"


def face_count_decision(face_count: int) -> FaceCountDecision:
    if face_count <= 0:
        return FaceCountDecision.NO_FACE
    if face_count == 1:
        return FaceCountDecision.FACE_OK
    return FaceCountDecision.MULTIPLE_FACES


@dataclass(frozen=True)
class DetectedFace:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    landmarks: np.ndarray  # shape (5, 2)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class DetectionResult:
    faces: tuple[DetectedFace, ...]
    inference_ms: float
    decision: FaceCountDecision

    @property
    def face_count(self) -> int:
        return len(self.faces)


class FaceDetector:
    """SCRFD detector wrapper; no HMAC, webcam, or liveness logic."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        det_thresh: float = DEFAULT_DET_THRESH,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"SCRFD model not found at {path}. See face_detection/models/MODEL_INFO.txt"
            )
        self._input_size = input_size
        self._det_thresh = det_thresh
        self._model_path = path
        self._scrfd = SCRFD(str(path))
        self._scrfd.prepare(-1, det_thresh=det_thresh)

    @property
    def model_path(self) -> Path:
        return self._model_path

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return DetectionResult((), 0.0, FaceCountDecision.NO_FACE)

        t0 = time.perf_counter_ns()
        bboxes, kpss = self._scrfd.detect(
            frame_bgr,
            input_size=self._input_size,
            det_thresh=self._det_thresh,
        )
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000

        faces: list[DetectedFace] = []
        if bboxes is not None and len(bboxes) > 0:
            for i, row in enumerate(bboxes):
                x1, y1, x2, y2, score = row.tolist()
                landmarks = (
                    kpss[i].copy()
                    if kpss is not None and len(kpss) > i
                    else np.zeros((5, 2), dtype=np.float32)
                )
                faces.append(
                    DetectedFace(x1, y1, x2, y2, float(score), landmarks)
                )

        decision = face_count_decision(len(faces))
        return DetectionResult(tuple(faces), elapsed_ms, decision)


def draw_detections(
    frame_bgr: np.ndarray,
    result: DetectionResult,
    *,
    detector_name: str = DETECTOR_NAME,
) -> np.ndarray:
    """Draw boxes, landmarks, and status overlay on a copy of the frame."""
    out = frame_bgr.copy()
    for face in result.faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{face.confidence:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        for px, py in face.landmarks:
            cv2.circle(out, (int(px), int(py)), 2, (0, 255, 255), -1)

    lines = [
        f"Detector: {detector_name}",
        f"Faces: {result.face_count}",
        f"Decision: {result.decision.value}",
        f"Detect: {result.inference_ms:.1f} ms",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            out,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 22
    return out

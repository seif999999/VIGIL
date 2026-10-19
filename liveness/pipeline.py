"""Gate SCRFD face-count policy before anti-spoof-mn3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from import_paths import prepare_liveness_imports

prepare_liveness_imports()

from detector import LivenessDetector, LivenessResult

# Face detection types imported at call sites to avoid circular imports.


@dataclass(frozen=True)
class PerceptionFrameResult:
    detection: object  # DetectionResult from face_detection.detector
    liveness: LivenessResult | None


def run_liveness_if_face_ok(
    frame_bgr: np.ndarray,
    detection_result,
    liveness: LivenessDetector | None,
) -> LivenessResult | None:
    """
    Run anti-spoof-mn3 only when exactly one face was detected (FACE_OK).
    """
    if liveness is None:
        return None
    decision = getattr(detection_result, "decision", None)
    if decision is None or getattr(decision, "value", decision) != "FACE_OK":
        return None
    if getattr(detection_result, "face_count", 0) != 1:
        return None
    face = detection_result.faces[0]
    return liveness.predict_from_frame_bbox(frame_bgr, face.bbox)

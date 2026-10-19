"""DeepFace FasNet (MiniFASNet ensemble) liveness — research/experimental path.

Uses DeepFace's public spoofing model API (deepface==0.0.102):
  MiniFASNetV2 @ crop scale 2.7 + MiniFASNetV1SE @ crop scale 4.0 → softmax sum →
  argmax class 1 = REAL.

SCRFD (or any) bbox is converted to (x, y, w, h) and passed with the **full frame**
so FasNet applies its native scaled crops. Do not feed a pre-cropped tight face
into analyze().

anti-spoof-mn3 remains the legacy baseline in detector.py.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

MODEL_NAME = "DeepFace_FasNet"
ENSEMBLE_NOTE = (
    "MiniFASNetV2 scale=2.7 + MiniFASNetV1SE scale=4.0; "
    "softmax sum; label 1=REAL (DeepFace 0.0.102 Fasnet.analyze)"
)


class FasNetClass(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"
    ERROR = "ERROR"


@dataclass(frozen=True)
class FasNetResult:
    is_real: bool | None
    antispoof_score: float | None
    predicted_class: FasNetClass
    inference_ms: float
    facial_area_xywh: tuple[int, int, int, int] | None
    error: str | None = None


def bbox_xyxy_to_xywh(
    bbox: tuple[float, float, float, float] | list[float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x = int(round(x1))
    y = int(round(y1))
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    return x, y, w, h


class FasNetLiveness:
    """Lazy-loaded DeepFace FasNet; reused across frames (no per-frame reload)."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._init_ms: float | None = None

    @property
    def init_ms(self) -> float | None:
        return self._init_ms

    def warm_up(self) -> None:
        self._ensure_model()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        t0 = time.perf_counter()
        from deepface.modules import modeling  # noqa: WPS433

        self._model = modeling.build_model(task="spoofing", model_name="Fasnet")
        self._init_ms = (time.perf_counter() - t0) * 1000.0
        return self._model

    def analyze_full_frame(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float] | list[float],
    ) -> FasNetResult:
        """Run FasNet on full BGR frame with SCRFD XYXY bbox → native (x,y,w,h) crops."""
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return FasNetResult(
                is_real=None,
                antispoof_score=None,
                predicted_class=FasNetClass.ERROR,
                inference_ms=0.0,
                facial_area_xywh=None,
                error="empty_frame",
            )
        xywh = bbox_xyxy_to_xywh(bbox_xyxy)
        t0 = time.perf_counter()
        try:
            model = self._ensure_model()
            is_real, score = model.analyze(img=frame_bgr, facial_area=xywh)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            real = bool(is_real)
            return FasNetResult(
                is_real=real,
                antispoof_score=float(score),
                predicted_class=FasNetClass.REAL if real else FasNetClass.SPOOF,
                inference_ms=elapsed_ms,
                facial_area_xywh=xywh,
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return FasNetResult(
                is_real=None,
                antispoof_score=None,
                predicted_class=FasNetClass.ERROR,
                inference_ms=elapsed_ms,
                facial_area_xywh=xywh,
                error=f"{type(exc).__name__}: {exc}",
            )

"""DeepFace FasNet (MiniFASNet ensemble) liveness — research/experimental path.

Uses DeepFace's public spoofing model API (deepface==0.0.102):
  MiniFASNetV2 @ crop scale 2.7 + MiniFASNetV1SE @ crop scale 4.0 → softmax sum →
  argmax class 1 = REAL.

Native DeepFace.analyze returns only (is_real, winning_class_score). This wrapper
performs the same ensemble forward pass once and also exposes averaged per-class
masses for research logging. Authentication still uses the native (is_real, score)
semantics identical to DeepFace:

  score = prediction[argmax] / 2
  is_real = (argmax == 1)

Class layout (Silent-Face / DeepFace): index 1 = REAL; indices 0 and 2 are the
non-real logits (both map to binary SPOOF). Averaged scores:

  class_scores_avg[i] = prediction[i] / 2   # sum over i ≈ 1.0
  real_score_avg = class_scores_avg[1]
  spoof_score_combined = class_scores_avg[0] + class_scores_avg[2]

These are uncalibrated ensemble masses, not verified probabilities.

Backends:
  - onnx (default when onnxruntime is available): DeepFace's official MiniFASNet
    ONNX graphs (softmax baked into the export). Same crops via FasNetUtils.crop.
  - tf (fallback / VIGIL_FASNET_BACKEND=tf): original Keras MiniFASNet path with
    explicit softmax — preserved for recoverability and parity checks.

SCRFD (or any) bbox is converted to (x, y, w, h) and passed with the **full frame**
so FasNet applies its native scaled crops. Do not feed a pre-cropped tight face
into analyze().

anti-spoof-mn3 remains the legacy baseline in detector.py.
"""

from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

MODEL_NAME = "DeepFace_FasNet"
ENSEMBLE_NOTE = (
    "MiniFASNetV2 scale=2.7 + MiniFASNetV1SE scale=4.0; "
    "softmax sum; label 1=REAL (DeepFace 0.0.102 Fasnet.analyze)"
)
# DeepFace / Silent-Face three-way output indices
CLASS_INDEX_REAL = 1

FasNetBackend = Literal["onnx", "tf"]

# Official DeepFace ONNX weight URLs (same as deepface.models.spoofing.onnx.FasNet).
_ONNX_V2_URLS = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/2.7_80x80_MiniFASNetV2.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/2.7_80x80_MiniFASNetV2.onnx",
]
_ONNX_V1SE_URLS = [
    "https://github.com/serengil/deepface_models/releases/download/v1.0/4_0_0_80x80_MiniFASNetV1SE.onnx",
    "https://huggingface.co/serengil/deepface/resolve/main/4_0_0_80x80_MiniFASNetV1SE.onnx",
]

_BACKEND_ENV = "VIGIL_FASNET_BACKEND"


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
    # Research logging only (same forward pass; not an auth API change)
    class_scores_avg: tuple[float, float, float] | None = None
    real_score_avg: float | None = None
    spoof_score_combined: float | None = None
    # Profiling only — does not affect decisions
    preprocess_ms: float | None = None
    mini_v2_ms: float | None = None
    mini_v1se_ms: float | None = None
    backend: str | None = None


def bbox_xyxy_to_xywh(
    bbox: tuple[float, float, float, float] | list[float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    x = int(round(x1))
    y = int(round(y1))
    w = max(1, int(round(x2 - x1)))
    h = max(1, int(round(y2 - y1)))
    return x, y, w, h


def scores_from_ensemble_prediction(
    prediction_row: np.ndarray,
) -> tuple[bool, float, tuple[float, float, float], float, float]:
    """
    Mirror DeepFace Fasnet.analyze decision math on a length-3 summed-softmax row.

    Returns:
      is_real, winning_score, class_scores_avg, real_score_avg, spoof_score_combined
    """
    pred = np.asarray(prediction_row, dtype=np.float64).reshape(-1)
    if pred.shape[0] != 3:
        raise ValueError(f"Expected 3-class ensemble row, got shape {pred.shape}")
    label = int(np.argmax(pred))
    is_real = label == CLASS_INDEX_REAL
    winning_score = float(pred[label] / 2.0)
    scores_avg = (
        float(pred[0] / 2.0),
        float(pred[1] / 2.0),
        float(pred[2] / 2.0),
    )
    real_avg = scores_avg[CLASS_INDEX_REAL]
    spoof_combined = scores_avg[0] + scores_avg[2]
    return is_real, winning_score, scores_avg, real_avg, spoof_combined


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


_crop_fn: Any | None = None


def _get_crop_fn() -> Any:
    global _crop_fn
    if _crop_fn is None:
        from deepface.models.spoofing.FasNetUtils import crop as _crop  # noqa: WPS433

        _crop_fn = _crop
    return _crop_fn


def resolve_fasnet_backend(requested: str | None = None) -> FasNetBackend:
    """
    Choose FasNet runtime.

    Priority: explicit arg → VIGIL_FASNET_BACKEND → onnx if importable → tf.
    """
    raw = (requested or os.environ.get(_BACKEND_ENV) or "").strip().lower()
    if raw in ("onnx", "tf"):
        if raw == "onnx":
            try:
                import onnxruntime  # noqa: F401, WPS433
            except ImportError as exc:
                raise ImportError(
                    "VIGIL_FASNET_BACKEND=onnx but onnxruntime is not installed"
                ) from exc
        return raw  # type: ignore[return-value]
    try:
        import onnxruntime  # noqa: F401, WPS433

        return "onnx"
    except ImportError:
        return "tf"


def _ensemble_prediction_row_tf(
    model: Any, img_bgr: np.ndarray, xywh: tuple[int, int, int, int]
) -> tuple[np.ndarray, float, float, float]:
    """Keras MiniFASNet path: logits → explicit softmax → sum (DeepFace TF analyze)."""
    crop = _get_crop_fn()
    x, y, w, h = xywh
    t_pre = time.perf_counter()
    first_img = crop(img_bgr, (x, y, w, h), 2.7, 80, 80)
    second_img = crop(img_bgr, (x, y, w, h), 4, 80, 80)
    first_img = np.expand_dims(first_img.astype(np.float32), axis=0)
    second_img = np.expand_dims(second_img.astype(np.float32), axis=0)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

    t1 = time.perf_counter()
    first_result = _softmax(np.asarray(model.first_model(first_img, training=False).numpy()))
    mini_v2_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    second_result = _softmax(np.asarray(model.second_model(second_img, training=False).numpy()))
    mini_v1se_ms = (time.perf_counter() - t2) * 1000.0

    prediction = first_result + second_result
    return (
        np.asarray(prediction[0], dtype=np.float64),
        preprocess_ms,
        mini_v2_ms,
        mini_v1se_ms,
    )


def _ensemble_prediction_row_onnx(
    sessions: tuple[Any, Any], img_bgr: np.ndarray, xywh: tuple[int, int, int, int]
) -> tuple[np.ndarray, float, float, float]:
    """
    Official DeepFace ONNX FasNet path.

    Softmax is baked into the exported graphs — do NOT apply _softmax again.
    Crops use the same FasNetUtils.crop helper as TF.
    """
    from deepface.commons.onnx_utils import run_session  # noqa: WPS433

    crop = _get_crop_fn()
    x, y, w, h = xywh
    first_session, second_session = sessions

    t_pre = time.perf_counter()
    first_img = crop(img_bgr, (x, y, w, h), 2.7, 80, 80)
    second_img = crop(img_bgr, (x, y, w, h), 4, 80, 80)
    preprocess_ms = (time.perf_counter() - t_pre) * 1000.0

    t1 = time.perf_counter()
    first_result = np.asarray(run_session(first_session, first_img), dtype=np.float64)
    mini_v2_ms = (time.perf_counter() - t1) * 1000.0

    t2 = time.perf_counter()
    second_result = np.asarray(run_session(second_session, second_img), dtype=np.float64)
    mini_v1se_ms = (time.perf_counter() - t2) * 1000.0

    prediction = np.asarray(first_result, dtype=np.float64) + np.asarray(
        second_result, dtype=np.float64
    )
    if prediction.ndim >= 2:
        row = prediction.reshape(prediction.shape[0], -1)[0]
    else:
        row = prediction.reshape(-1)
    return np.asarray(row[:3], dtype=np.float64), preprocess_ms, mini_v2_ms, mini_v1se_ms


class FasNetLiveness:
    """Lazy-loaded FasNet ensemble; reused across frames (no per-frame reload)."""

    def __init__(self, *, backend: str | None = None) -> None:
        self._backend: FasNetBackend = resolve_fasnet_backend(backend)
        self._model: Any | None = None
        self._onnx_sessions: tuple[Any, Any] | None = None
        self._init_ms: float | None = None

    @property
    def init_ms(self) -> float | None:
        return self._init_ms

    @property
    def backend(self) -> FasNetBackend:
        return self._backend

    def warm_up(self) -> None:
        self._ensure_model()

    def _ensure_model(self) -> FasNetBackend:
        if self._backend == "onnx":
            if self._onnx_sessions is not None:
                return self._backend
            t0 = time.perf_counter()
            from deepface.commons.onnx_utils import build_session  # noqa: WPS433

            _get_crop_fn()
            first = build_session(
                file_name="2.7_80x80_MiniFASNetV2.onnx", source_url=_ONNX_V2_URLS
            )
            second = build_session(
                file_name="4_0_0_80x80_MiniFASNetV1SE.onnx", source_url=_ONNX_V1SE_URLS
            )
            self._onnx_sessions = (first, second)
            self._init_ms = (time.perf_counter() - t0) * 1000.0
            return self._backend

        if self._model is not None:
            return self._backend
        t0 = time.perf_counter()
        try:
            from live_runtime import configure_tf_threads_for_inference  # noqa: WPS433

            configure_tf_threads_for_inference()
        except Exception:  # noqa: BLE001
            pass
        from deepface.modules import modeling  # noqa: WPS433

        self._model = modeling.build_model(task="spoofing", model_name="Fasnet")
        _get_crop_fn()
        self._init_ms = (time.perf_counter() - t0) * 1000.0
        return self._backend

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
                backend=self._backend,
            )
        xywh = bbox_xyxy_to_xywh(bbox_xyxy)
        t0 = time.perf_counter()
        try:
            backend = self._ensure_model()
            if backend == "onnx":
                assert self._onnx_sessions is not None
                pred_row, preprocess_ms, mini_v2_ms, mini_v1se_ms = (
                    _ensemble_prediction_row_onnx(
                        self._onnx_sessions, frame_bgr, xywh
                    )
                )
            else:
                assert self._model is not None
                pred_row, preprocess_ms, mini_v2_ms, mini_v1se_ms = (
                    _ensemble_prediction_row_tf(self._model, frame_bgr, xywh)
                )
            is_real, score, scores_avg, real_avg, spoof_combined = (
                scores_from_ensemble_prediction(pred_row)
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return FasNetResult(
                is_real=bool(is_real),
                antispoof_score=float(score),
                predicted_class=FasNetClass.REAL if is_real else FasNetClass.SPOOF,
                inference_ms=elapsed_ms,
                facial_area_xywh=xywh,
                error=None,
                class_scores_avg=scores_avg,
                real_score_avg=real_avg,
                spoof_score_combined=spoof_combined,
                preprocess_ms=preprocess_ms,
                mini_v2_ms=mini_v2_ms,
                mini_v1se_ms=mini_v1se_ms,
                backend=backend,
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
                backend=self._backend,
            )

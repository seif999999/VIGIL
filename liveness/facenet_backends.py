"""FaceNet-128 inference backends: TensorFlow (DeepFace) and same-weight ONNX.

Shared preprocessing mirrors deepface==0.0.102 representation.represent for:
  detector_backend=opencv, align=True, normalization=base, Facenet 160x160.

Does not change enrollment data or the 9.0 threshold.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

MODEL_NAME = "Facenet"
EMBEDDING_DIM = 128
DEFAULT_ONNX_PATH = (
    Path(__file__).resolve().parent / "data" / "facenet_onnx" / "facenet128_deepface_vigil.onnx"
)

FaceNetBackend = Literal["tf", "onnx"]
_BACKEND_ENV = "VIGIL_FACENET_BACKEND"


@dataclass(frozen=True)
class FaceNetStageTiming:
    extract_faces_ms: float
    rgb_to_bgr_ms: float
    resize_pad_ms: float
    normalize_ms: float
    forward_ms: float
    total_ms: float
    backend: str


def resolve_facenet_backend(requested: str | None = None) -> FaceNetBackend:
    """
    Choose FaceNet runtime.

    Priority: explicit arg → VIGIL_FACENET_BACKEND → onnx when same-weight
    export + onnxruntime exist → tf. Force with tf|onnx|auto.
    """
    raw = (requested or os.environ.get(_BACKEND_ENV) or "").strip().lower()
    if raw in ("tf", "onnx"):
        if raw == "onnx":
            try:
                import onnxruntime  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "VIGIL_FACENET_BACKEND=onnx but onnxruntime is not installed"
                ) from exc
            if not DEFAULT_ONNX_PATH.is_file():
                raise FileNotFoundError(
                    f"VIGIL_FACENET_BACKEND=onnx but missing {DEFAULT_ONNX_PATH}. "
                    "Run: python convert_facenet_to_onnx.py"
                )
        return raw  # type: ignore[return-value]
    # Default / auto: same-weight ONNX when export + ORT exist (parity-verified).
    if DEFAULT_ONNX_PATH.is_file():
        try:
            import onnxruntime  # noqa: F401

            return "onnx"
        except ImportError:
            return "tf"
    return "tf"


def preprocess_facenet_input(
    image_bgr: np.ndarray,
    *,
    enforce_detection: bool = False,
    align: bool = True,
) -> tuple[np.ndarray, FaceNetStageTiming]:
    """Return (1,160,160,3) float32 model input in [0,1] + stage timings."""
    from deepface.modules import detection, preprocessing

    t_all = time.perf_counter()

    t0 = time.perf_counter()
    img_objs = detection.extract_faces(
        img_path=image_bgr,
        detector_backend="opencv",
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=0,
        anti_spoofing=False,
        max_faces=1,
    )
    extract_ms = (time.perf_counter() - t0) * 1000.0
    if not img_objs:
        raise RuntimeError("FaceNet preprocess: no face from opencv extract_faces")

    t0 = time.perf_counter()
    face_bgr = img_objs[0]["face"][:, :, ::-1]
    rgb_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    batch = preprocessing.resize_image(img=face_bgr, target_size=(160, 160))
    resize_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    batch = preprocessing.normalize_input(img=batch, normalization="base")
    norm_ms = (time.perf_counter() - t0) * 1000.0

    timing = FaceNetStageTiming(
        extract_faces_ms=extract_ms,
        rgb_to_bgr_ms=rgb_ms,
        resize_pad_ms=resize_ms,
        normalize_ms=norm_ms,
        forward_ms=0.0,
        total_ms=(time.perf_counter() - t_all) * 1000.0,
        backend="preprocess",
    )
    return np.asarray(batch, dtype=np.float32), timing


class TfFaceNetForward:
    """DeepFace-cached Keras FaceNet-128 forward."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._init_ms: float | None = None

    @property
    def init_ms(self) -> float | None:
        return self._init_ms

    def warm_up(self) -> None:
        if self._client is not None:
            return
        t0 = time.perf_counter()
        from deepface.modules import modeling

        self._client = modeling.build_model(
            task="facial_recognition", model_name=MODEL_NAME
        )
        # Touch graph once
        self._client.forward(np.zeros((1, 160, 160, 3), dtype=np.float32))
        self._init_ms = (time.perf_counter() - t0) * 1000.0

    def forward(self, batch: np.ndarray) -> tuple[list[float], float]:
        self.warm_up()
        assert self._client is not None
        t0 = time.perf_counter()
        emb = self._client.forward(batch)
        ms = (time.perf_counter() - t0) * 1000.0
        if emb and isinstance(emb[0], list):
            emb = emb[0]
        return list(emb), ms


class OnnxFaceNetForward:
    """Same-weight ONNX Runtime FaceNet-128 (exported from DeepFace Keras)."""

    def __init__(self, onnx_path: Path | None = None) -> None:
        self.onnx_path = Path(onnx_path) if onnx_path else DEFAULT_ONNX_PATH
        self._session: Any | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._init_ms: float | None = None

    @property
    def init_ms(self) -> float | None:
        return self._init_ms

    def warm_up(self) -> None:
        if self._session is not None:
            return
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"FaceNet ONNX not found at {self.onnx_path}. "
                "Run: python convert_facenet_to_onnx.py"
            )
        t0 = time.perf_counter()
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Leave headroom for SCRFD ORT + FasNet ORT + camera.
        cpus = os.cpu_count() or 4
        so.intra_op_num_threads = max(1, min(4, cpus - 1))
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self.forward(np.zeros((1, 160, 160, 3), dtype=np.float32))
        self._init_ms = (time.perf_counter() - t0) * 1000.0

    def forward(self, batch: np.ndarray) -> tuple[list[float], float]:
        self.warm_up()
        assert self._session is not None and self._input_name and self._output_name
        x = np.ascontiguousarray(batch, dtype=np.float32)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        t0 = time.perf_counter()
        out = self._session.run([self._output_name], {self._input_name: x})[0]
        ms = (time.perf_counter() - t0) * 1000.0
        row = np.asarray(out, dtype=np.float64).reshape(-1)
        if row.shape[0] != EMBEDDING_DIM:
            raise ValueError(f"unexpected onnx embedding dim={row.shape[0]}")
        return row.tolist(), ms


def embed_bgr_with_backend(
    image_bgr: np.ndarray,
    *,
    backend: FaceNetBackend,
    enforce_detection: bool = False,
    tf_fwd: TfFaceNetForward | None = None,
    onnx_fwd: OnnxFaceNetForward | None = None,
) -> tuple[list[float] | None, FaceNetStageTiming, str | None]:
    """Preprocess once, run TF or ONNX forward. Fail-closed on errors."""
    t_all = time.perf_counter()
    try:
        batch, pre = preprocess_facenet_input(
            image_bgr, enforce_detection=enforce_detection
        )
        if backend == "onnx":
            engine = onnx_fwd or OnnxFaceNetForward()
            emb, fwd_ms = engine.forward(batch)
        else:
            engine = tf_fwd or TfFaceNetForward()
            emb, fwd_ms = engine.forward(batch)
        timing = FaceNetStageTiming(
            extract_faces_ms=pre.extract_faces_ms,
            rgb_to_bgr_ms=pre.rgb_to_bgr_ms,
            resize_pad_ms=pre.resize_pad_ms,
            normalize_ms=pre.normalize_ms,
            forward_ms=fwd_ms,
            total_ms=(time.perf_counter() - t_all) * 1000.0,
            backend=backend,
        )
        return emb, timing, None
    except Exception as exc:  # noqa: BLE001
        timing = FaceNetStageTiming(
            extract_faces_ms=0.0,
            rgb_to_bgr_ms=0.0,
            resize_pad_ms=0.0,
            normalize_ms=0.0,
            forward_ms=0.0,
            total_ms=(time.perf_counter() - t_all) * 1000.0,
            backend=backend,
        )
        return None, timing, f"{type(exc).__name__}: {exc}"

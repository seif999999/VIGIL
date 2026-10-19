"""Research-only IADG (CVPR 2023) face anti-spoofing inference.

Upstream: https://github.com/qianyuzqy/IADG
Paper: Instance-Aware Domain Generalization for Face Anti-Spoofing (CVPR 2023)

Official test scoring (test.py):
  live_score = softmax(cls_logits)[:, 0] + mean(depth_map)
  Higher live_score => more live (label 0 = real; metric.roc_curve pos_label=0).

Official preprocess (configs + DG_dataset + transforms):
  - Crop face from 5 landmarks with margin (default 0.7), or bbox with same margin
  - BGR -> RGB
  - Resize to image_size (256)
  - Normalize mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
  - NCHW float

Checkpoint used: OCI2M leave-one-out (train Oulu/CASIA/Idiap, test MSU) model_best.
Not connected to production authentication.
"""

from __future__ import annotations

import enum
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
IADG_ROOT = Path(__file__).resolve().parent / "iadg"
DEFAULT_CHECKPOINT = BASE_DIR / "models" / "research" / "iadg" / "OCI2M_model_best.pth.tar"
MODEL_NAME = "IADG_OCI2M"
DEFAULT_MARGIN = 0.7
DEFAULT_IMAGE_SIZE = 256
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


class PredictedClass(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"


@dataclass(frozen=True)
class IadgResult:
    live_score: float
    cls_live_prob: float
    depth_mean: float
    spoof_cls_prob: float
    predicted_class: PredictedClass
    argmax_cls: int
    inference_ms: float
    crop_mode: str


def crop_face_from_landmarks(
    frame_bgr: np.ndarray,
    landmarks_5x2: np.ndarray,
    *,
    margin: float = DEFAULT_MARGIN,
) -> np.ndarray | None:
    """Official _crop_from_5landmarks with margin (square side from landmark span)."""
    if frame_bgr is None or landmarks_5x2 is None:
        return None
    pts = np.asarray(landmarks_5x2, dtype=np.float32).reshape(5, 2)
    x_list = pts[:, 0]
    y_list = pts[:, 1]
    x, y = float(np.min(x_list)), float(np.min(y_list))
    w, h = float(np.max(x_list) - x), float(np.max(y_list) - y)
    side = w if w > h else h
    if side <= 1:
        return None
    x_m = int(side * margin / 2)
    y_m = int(side * margin / 2)
    x1 = int(x) - x_m
    y1 = int(y) - y_m
    x2 = int(x) + int(side) + x_m
    y2 = int(y) + int(side) + y_m
    max_h, max_w = frame_bgr.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(max_w, x2)
    y2 = min(max_h, y2)
    if x1 >= x2 or y1 >= y2:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size else None


def crop_face_from_bbox_xyxy(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    margin: float = DEFAULT_MARGIN,
) -> np.ndarray | None:
    """Official _crop_face_from_bbox style: xywh-like square + margin."""
    x1, y1, x2, y2 = bbox_xyxy
    x, y = float(x1), float(y1)
    w, h = float(x2 - x1), float(y2 - y1)
    side = w if w > h else h
    if side <= 1:
        return None
    x_m = int(side * margin / 2)
    y_m = int(side * margin / 2)
    xa = int(x) - x_m
    ya = int(y) - y_m
    xb = int(x) + int(side) + x_m
    yb = int(y) + int(side) + y_m
    max_h, max_w = frame_bgr.shape[:2]
    xa = max(0, xa)
    ya = max(0, ya)
    xb = min(max_w, xb)
    yb = min(max_h, yb)
    if xa >= xb or ya >= yb:
        return None
    crop = frame_bgr[ya:yb, xa:xb]
    return crop if crop.size else None


def preprocess_face_bgr(
    face_bgr: np.ndarray,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    mean: tuple[float, float, float] = DEFAULT_MEAN,
    std: tuple[float, float, float] = DEFAULT_STD,
) -> torch.Tensor:
    """BGR face crop -> RGB NCHW normalized tensor (albumentations-equivalent)."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    mean_a = np.asarray(mean, dtype=np.float32)
    std_a = np.asarray(std, dtype=np.float32)
    norm = (arr - mean_a) / std_a
    return torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0)


class IadgResearch:
    """CPU inference for official IADG OCI2M checkpoint."""

    def __init__(self, checkpoint: Path | str | None = None) -> None:
        path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not path.is_file():
            raise FileNotFoundError(
                f"IADG checkpoint not found at {path}. See models/research/iadg/MODEL_INFO.txt"
            )
        # Upstream imports `utils.*` / `models.*` relative to IADG root.
        root = str(IADG_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)

        from models.framework import Framework  # noqa: WPS433

        self._checkpoint = path
        self._device = torch.device("cpu")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        args = raw["args"]
        params = dict(args.model.params)
        # Checkpoint may include training-only keys (e.g. clusters).
        allowed = {
            "total_dkg_flag",
            "style_dim",
            "base_style_num",
            "concentration_coeff",
            "in_channels",
            "mid_channels",
            "model_initial",
        }
        params = {k: v for k, v in params.items() if k in allowed}
        self._image_size = int(args.transform.image_size)
        self._mean = tuple(float(x) for x in args.transform.mean)
        self._std = tuple(float(x) for x in args.transform.std)
        self._model = Framework(**params).to(self._device)
        self._model.load_state_dict(raw["state_dict"], strict=True)
        self._model.eval()
        self._meta = {
            "protocol": "OCI2M",
            "epoch": int(raw.get("epoch")) if "epoch" in raw else None,
            "metric": str(raw.get("metric")) if "metric" in raw else None,
            "params": params,
            "image_size": self._image_size,
            "mean": list(self._mean),
            "std": list(self._std),
            "score_rule": "softmax(cls)[:,0] + mean(depth); higher => LIVE (label 0)",
        }

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def predict_face_bgr(self, face_bgr: np.ndarray, *, crop_mode: str) -> IadgResult:
        tensor = preprocess_face_bgr(
            face_bgr,
            image_size=self._image_size,
            mean=self._mean,
            std=self._std,
        ).to(self._device)
        # Dummy labels unused in eval path (Shader bypassed when not training).
        labels = torch.zeros(1, dtype=torch.long, device=self._device)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            outputs_catcls, outputs_catdepth, *_ = self._model(
                tensor, labels, apply_shade=True, cal_covstat=False, apply_wt=False
            )
            cls_logits = outputs_catcls["out"]
            cls_prob = F.softmax(cls_logits, dim=1).cpu().numpy().reshape(-1)
            depth = outputs_catdepth["out"]
            depth_mean = float(depth.reshape(depth.shape[0], -1).mean(dim=1)[0].item())
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        cls_live = float(cls_prob[0])
        cls_spoof = float(cls_prob[1])
        live_score = cls_live + depth_mean
        argmax = int(np.argmax(cls_prob))
        predicted = PredictedClass.REAL if argmax == 0 else PredictedClass.SPOOF
        return IadgResult(
            live_score=float(live_score),
            cls_live_prob=cls_live,
            depth_mean=depth_mean,
            spoof_cls_prob=cls_spoof,
            predicted_class=predicted,
            argmax_cls=argmax,
            inference_ms=elapsed_ms,
            crop_mode=crop_mode,
        )

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
        landmarks_5x2: np.ndarray | None,
        *,
        margin: float = DEFAULT_MARGIN,
        prefer_landmarks: bool = True,
    ) -> IadgResult | None:
        crop = None
        mode = "none"
        if prefer_landmarks and landmarks_5x2 is not None:
            crop = crop_face_from_landmarks(frame_bgr, landmarks_5x2, margin=margin)
            mode = "landmarks_margin"
        if crop is None:
            crop = crop_face_from_bbox_xyxy(frame_bgr, bbox_xyxy, margin=margin)
            mode = "bbox_margin"
        if crop is None:
            return None
        return self.predict_face_bgr(crop, crop_mode=mode)

"""Research-only CVPR 2024 Joint Physical-Digital FAS (MobileNetV3-Small).

Source: https://github.com/Xianhua-He/cvpr2024-face-anti-spoofing-challenge
Paper: Joint Physical-Digital Facial Attack Detection Via Simulating Spoofing Clues

Official notes (README):
  - Label mapping: live=0, spoof=1
  - General-scenario lightweight models recommend full-image input
  - Val preprocess (datasets/utils.get_val_dataset, default): Resize(input_size),
    BGR->RGB, ImageNet Normalize, ToTensor
  - Export script uses input_size 224 and EMA weights when present

Not used in production authentication.
"""

from __future__ import annotations

import enum
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from research.cvpr2024_mobilenetv3_arch import mobilenet_v3_small

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = BASE_DIR / "models" / "research" / "mobilenet_v3_small.pth"
MODEL_NAME = "cvpr2024_mobilenet_v3_small"
INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Official: live=0, spoof=1
CLASS_INDEX_LIVE = 0
CLASS_INDEX_SPOOF = 1


class PredictedClass(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"


@dataclass(frozen=True)
class CvprFasResult:
    live_prob: float
    spoof_prob: float
    predicted_class: PredictedClass
    argmax_label: int
    inference_ms: float
    input_mode: str


def preprocess_bgr_to_nchw(image_bgr: np.ndarray, size: int = INPUT_SIZE) -> torch.Tensor:
    """Official val path without albumentations: Resize, BGR2RGB, /255, ImageNet norm."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image")
    resized = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(norm, (2, 0, 1))
    return torch.from_numpy(chw).unsqueeze(0)


def _strip_module_prefix(state_dict: dict) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        name = key
        if name.startswith("module.module."):
            name = name[len("module.module.") :]
        elif name.startswith("module."):
            name = name[len("module.") :]
        out[name] = value
    return out


class Cvpr2024MobileNetV3Research:
    """CPU inference for official general-scenario MobileNetV3-Small weights."""

    def __init__(self, checkpoint: Path | str | None = None, *, use_ema: bool = True) -> None:
        path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not path.is_file():
            raise FileNotFoundError(
                f"CVPR2024 MobileNetV3 checkpoint not found at {path}. "
                "See models/research/MODEL_INFO_cvpr2024.txt"
            )
        self._checkpoint = path
        self._device = torch.device("cpu")
        self._model = mobilenet_v3_small(pretrained=False, num_classes=2).to(self._device)
        raw = torch.load(path, map_location=self._device, weights_only=False)
        if use_ema and isinstance(raw, dict) and raw.get("state_dict_ema") is not None:
            state = raw["state_dict_ema"]
            self._weight_source = "state_dict_ema"
        elif isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
            self._weight_source = "state_dict"
        else:
            state = raw
            self._weight_source = "raw"
        self._model.load_state_dict(_strip_module_prefix(state), strict=True)
        self._model.eval()
        self._meta = {
            "epoch": int(raw.get("epoch")) if isinstance(raw, dict) and "epoch" in raw else None,
            "arch": raw.get("arch") if isinstance(raw, dict) else None,
            "weight_source": self._weight_source,
        }

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def predict_bgr(self, image_bgr: np.ndarray, *, input_mode: str = "full_frame") -> CvprFasResult:
        tensor = preprocess_bgr_to_nchw(image_bgr).to(self._device)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        live_p = float(probs[CLASS_INDEX_LIVE])
        spoof_p = float(probs[CLASS_INDEX_SPOOF])
        argmax = int(np.argmax(probs))
        predicted = PredictedClass.REAL if argmax == CLASS_INDEX_LIVE else PredictedClass.SPOOF
        return CvprFasResult(
            live_prob=live_p,
            spoof_prob=spoof_p,
            predicted_class=predicted,
            argmax_label=argmax,
            inference_ms=elapsed_ms,
            input_mode=input_mode,
        )

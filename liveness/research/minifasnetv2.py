"""Research-only MiniFASNetV2 (Silent-Face-Anti-Spoofing) inference."""

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

from research.silent_face_minifasnet import MiniFASNetV2

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = BASE_DIR / "models" / "research" / "2.7_80x80_MiniFASNetV2.pth"
MODEL_NAME = "2.7_80x80_MiniFASNetV2"
CROP_SCALE = 2.7
INPUT_SIZE = (80, 80)  # (h, w) per parse_model_name on checkpoint filename

# Official demo (test.py): argmax class index 1 => Real Face; 0 and 2 => Fake Face.
CLASS_INDEX_REAL = 1
NUM_CLASSES = 3


class PredictedClass(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"


@dataclass(frozen=True)
class MiniFASResult:
    softmax: tuple[float, float, float]
    real_class_prob: float
    spoof_mass: float
    predicted_class: PredictedClass
    argmax_label: int
    inference_ms: float


def parse_model_name(model_name: str) -> tuple[int, int, str, float | None]:
    """Silent-Face utility.parse_model_name (filename without path)."""
    info = model_name.split("_")[0:-1]
    h_input, w_input = info[-1].split("x")
    model_type = model_name.split(".pth")[0].split("_")[-1]
    scale = None if info[0] == "org" else float(info[0])
    return int(h_input), int(w_input), model_type, scale


def get_kernel(height: int, width: int) -> tuple[int, int]:
    return (height + 15) // 16, (width + 15) // 16


def scrfd_xyxy_to_silent_bbox(bbox_xyxy: tuple[float, float, float, float]) -> list[int]:
    """SCRFD (x1,y1,x2,y2) -> Silent-Face [left, top, width, height]."""
    x1, y1, x2, y2 = bbox_xyxy
    return [int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))]


class _CropImage:
    """Vendored from Silent-Face generate_patches.CropImage."""

    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox[0], bbox[1], bbox[2], bbox[3]
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y
        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1
        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):
        if not crop:
            return cv2.resize(org_img, (out_w, out_h))
        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(
            src_w, src_h, bbox, scale
        )
        img = org_img[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
        return cv2.resize(img, (out_w, out_h))


_cropper = _CropImage()


def crop_face_for_minifas(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    scale: float = CROP_SCALE,
    out_w: int = INPUT_SIZE[1],
    out_h: int = INPUT_SIZE[0],
) -> np.ndarray | None:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    bbox = scrfd_xyxy_to_silent_bbox(bbox_xyxy)
    if bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return _cropper.crop(frame_bgr, bbox, scale, out_w, out_h, crop=True)


def to_tensor_bgr_uint8(crop_bgr: np.ndarray) -> torch.Tensor:
    """Silent-Face ToTensor on ndarray: HWC BGR uint8 -> CHW float (0–255, not /255)."""
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        raise ValueError("Expected BGR image")
    return torch.from_numpy(crop_bgr.transpose((2, 0, 1))).float()


class MiniFASNetV2Research:
    """Single-checkpoint MiniFASNetV2; CPU by default."""

    def __init__(self, checkpoint: Path | str | None = None) -> None:
        path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
        if not path.is_file():
            raise FileNotFoundError(
                f"MiniFASNetV2 checkpoint not found at {path}. See models/research/MODEL_INFO.txt"
            )
        self._checkpoint = path
        name = path.name
        h_in, w_in, model_type, scale = parse_model_name(name)
        if model_type != "MiniFASNetV2" or scale != CROP_SCALE or (h_in, w_in) != INPUT_SIZE:
            raise RuntimeError(f"Unexpected checkpoint metadata from {name}: {(h_in, w_in, model_type, scale)}")
        self._out_h, self._out_w = h_in, w_in
        self._scale = scale
        kernel = get_kernel(h_in, w_in)
        self._device = torch.device("cpu")
        self._model = MiniFASNetV2(conv6_kernel=kernel).to(self._device)
        state_dict = torch.load(path, map_location=self._device, weights_only=True)
        first_key = next(iter(state_dict))
        if first_key.find("module.") >= 0:
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                new_state_dict[key[7:]] = value
            state_dict = new_state_dict
        self._model.load_state_dict(state_dict)
        self._model.eval()

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint

    def predict_crop_bgr(self, crop_bgr: np.ndarray) -> MiniFASResult:
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty crop")
        tensor = to_tensor_bgr_uint8(crop_bgr).unsqueeze(0).to(self._device)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            logits = self._model.forward(tensor)
            prob = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000
        if prob.shape[0] != NUM_CLASSES:
            raise RuntimeError(f"Expected {NUM_CLASSES} classes, got {prob.shape}")
        p0, p1, p2 = float(prob[0]), float(prob[1]), float(prob[2])
        argmax = int(np.argmax(prob))
        predicted = PredictedClass.REAL if argmax == CLASS_INDEX_REAL else PredictedClass.SPOOF
        return MiniFASResult(
            softmax=(p0, p1, p2),
            real_class_prob=p1,
            spoof_mass=p0 + p2,
            predicted_class=predicted,
            argmax_label=argmax,
            inference_ms=elapsed_ms,
        )

    def predict_frame_bbox(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> MiniFASResult | None:
        crop = crop_face_for_minifas(frame_bgr, bbox_xyxy, scale=self._scale, out_w=self._out_w, out_h=self._out_h)
        if crop is None:
            return None
        return self.predict_crop_bgr(crop)

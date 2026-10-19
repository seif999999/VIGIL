"""Research-only FSFM (CVPR 2025) Face Anti-Spoofing inference.

Upstream:
  Paper: FSFM: A Generalizable Face Security Foundation Model (CVPR 2025)
  Repo:  https://github.com/wolo-wolo/FSFM-CVPR25
  Hub:   https://huggingface.co/Wolowolo/fsfm-3c

Checkpoint choice (documented before benchmarking — not cherry-picked by our metrics):
  finetuned_models/MCIO_protocol/Both_MCIO/checkpoint-min_val_loss.pth
  Official demo FAS weights: ViT-B/16 fine-tuned on all MCIO domains
  (MSU, CASIA, Idiap/Replay, Oulu), 2 frames/video, with a 2-class head.
  Alternatives exist (LOO OCI2M/OMI2C/OCM2I/ICM2O fas_model_avg .pth.tar);
  Both_MCIO is the released multi-domain FAS checkpoint used by the authors'
  Hugging Face Space for FAS, so it is the fair general-transfer candidate.

Official scoring (engine_finetune.test_two_class + Space FAS branch):
  live_prob = softmax(logits)[:, 1]
  Bonafide if live_prob > 0.5 else Spoofing  (class 1 = Bonafide/real)

Official preprocess (Space FAS + build_transform eval path):
  - Square face crop with ~1.3x bbox expansion (dlib demo); we use SCRFD bbox
  - RGB, Resize 224 BICUBIC, ToTensor, Normalize from pretrain_ds_mean_std.txt
    mean=[0.5482, 0.4234, 0.3655], std=[0.2789, 0.2439, 0.2349]
  - NOT ImageNet mean/std

License: CC BY-NC 4.0 (research / non-commercial only).
Not connected to production authentication.
"""

from __future__ import annotations

import enum
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
FSFM_CODE = Path(__file__).resolve().parent / "fsfm"
DEFAULT_CHECKPOINT = (
    BASE_DIR
    / "models"
    / "research"
    / "fsfm"
    / "Both_MCIO_checkpoint-min_val_loss.pth"
)
DEFAULT_MEAN_STD = (
    BASE_DIR / "models" / "research" / "fsfm" / "pretrain_ds_mean_std.txt"
)
MODEL_NAME = "FSFM_Both_MCIO_ViT-B"
HF_REPO = "Wolowolo/fsfm-3c"
HF_CKPT_REL = "finetuned_models/MCIO_protocol/Both_MCIO/checkpoint-min_val_loss.pth"
DEFAULT_INPUT_SIZE = 224
DEFAULT_BBOX_EXPAND = 1.3
DEFAULT_DROP_PATH = 0.1


class PredictedClass(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"


@dataclass(frozen=True)
class FsfmResult:
    live_prob: float
    spoof_prob: float
    predicted_class: PredictedClass
    argmax_cls: int
    inference_ms: float
    crop_mode: str
    device: str


def _load_mean_std(path: Path) -> tuple[list[float], list[float]]:
    line = path.read_text(encoding="utf-8").splitlines()[0]
    stat = json.loads(line)
    return list(stat["mean"]), list(stat["std"])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def crop_face_square_expand(
    frame_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    expand: float = DEFAULT_BBOX_EXPAND,
) -> np.ndarray | None:
    """Official Space get_boundingbox-style square crop with expand factor."""
    if frame_bgr is None:
        return None
    x1, y1, x2, y2 = map(float, bbox_xyxy)
    max_h, max_w = frame_bgr.shape[:2]
    size_bb = int(max(x2 - x1, y2 - y1) * expand)
    if size_bb < 2:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    xa = max(int(cx - size_bb // 2), 0)
    ya = max(int(cy - size_bb // 2), 0)
    size_bb = min(max_w - xa, size_bb)
    size_bb = min(max_h - ya, size_bb)
    if size_bb < 2:
        return None
    crop = frame_bgr[ya : ya + size_bb, xa : xa + size_bb]
    return crop if crop.size else None


class FsfmResearch:
    """Isolated FSFM Both_MCIO FAS wrapper (research only)."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        mean_std_path: Path | None = None,
        *,
        device: str | None = None,
        bbox_expand: float = DEFAULT_BBOX_EXPAND,
    ) -> None:
        if str(FSFM_CODE) not in sys.path:
            sys.path.insert(0, str(FSFM_CODE))
        import models_vit  # noqa: WPS433 — local upstream copy

        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.mean_std_path = Path(mean_std_path or DEFAULT_MEAN_STD)
        self.bbox_expand = float(bbox_expand)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"FSFM checkpoint missing: {self.checkpoint_path}. "
                f"Download {HF_CKPT_REL} from {HF_REPO}."
            )
        if not self.mean_std_path.is_file():
            raise FileNotFoundError(f"Missing mean/std file: {self.mean_std_path}")

        mean, std = _load_mean_std(self.mean_std_path)
        self.mean = mean
        self.std = std
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = models_vit.__dict__["vit_base_patch16"](
            num_classes=2,
            drop_path_rate=DEFAULT_DROP_PATH,
            global_pool=True,
        )
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.meta = {
            "checkpoint": str(self.checkpoint_path),
            "sha256": self.checkpoint_sha256,
            "hf_repo": HF_REPO,
            "hf_path": HF_CKPT_REL,
            "protocol": "Both_MCIO",
            "architecture": "vit_base_patch16",
            "num_classes": 2,
            "global_pool": True,
            "drop_path": DEFAULT_DROP_PATH,
            "input_size": DEFAULT_INPUT_SIZE,
            "bbox_expand": self.bbox_expand,
            "mean": self.mean,
            "std": self.std,
            "class_mapping": "0=SPOOF/fake, 1=REAL/bonafide",
            "score_rule": "softmax[:,1] = live_prob (Bonafide)",
            "license": "CC BY-NC 4.0",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "device": str(self.device),
        }
        self.model.to(self.device)
        self.model.eval()

    def preprocess_bgr_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE), interpolation=cv2.INTER_CUBIC
        )
        x = rgb.astype(np.float32) / 255.0
        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x).unsqueeze(0)

    @torch.inference_mode()
    def predict_crop(self, crop_bgr: np.ndarray, *, crop_mode: str = "bbox_expand") -> FsfmResult:
        t0 = time.perf_counter()
        tensor = self.preprocess_bgr_crop(crop_bgr).to(self.device, non_blocking=True)
        # Call forward_features + head directly: newer timm.forward() pool APIs
        # diverge from FSFM's VisionTransformer subclass (global_pool=True).
        feats = self.model.forward_features(tensor)
        logits = self.model.head(feats)
        probs = F.softmax(logits, dim=1)[0]
        spoof_p = float(probs[0].item())
        live_p = float(probs[1].item())
        argmax = int(torch.argmax(probs).item())
        pred = PredictedClass.REAL if argmax == 1 else PredictedClass.SPOOF
        ms = (time.perf_counter() - t0) * 1000.0
        return FsfmResult(
            live_prob=live_p,
            spoof_prob=spoof_p,
            predicted_class=pred,
            argmax_cls=argmax,
            inference_ms=ms,
            crop_mode=crop_mode,
            device=str(self.device),
        )

    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float],
    ) -> FsfmResult | None:
        crop = crop_face_square_expand(frame_bgr, bbox_xyxy, expand=self.bbox_expand)
        if crop is None:
            return None
        return self.predict_crop(crop, crop_mode=f"bbox_expand_{self.bbox_expand}")

    def peak_cuda_memory_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))

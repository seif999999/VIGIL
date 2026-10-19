"""Research-only FLIP-MCL (ICCV 2023) face anti-spoofing inference.

Upstream: https://github.com/koushiksrivats/FLIP
Paper: FLIP: Cross-domain Face Anti-spoofing with Language Guidance (ICCV 2023)

Checkpoint choice (documented BEFORE benchmarking — not cherry-picked by our metrics):
  Official model zoo MCIO leave-out **MSU 0-shot** SharePoint link resolves to
  `msu_flip_mcl.pth.tar` (FLIP-MCL, not FLIP-V).
  Protocol: train Oulu+CASIA+Replay (+Celeb frames in loader), leave-out MSU
  (same OCI→M zero-shot protocol used previously for IADG / FSFM LOO).
  Prefer 0-shot over 5-shot (no target-domain adaptation). Other LOO domains
  (CASIA/IDIAP/OULU) exist; MSU leave-out is the conventional first protocol
  entry and is not selected from VIGIL Camo/PRINT outcomes.

Scoring (utils/evaluate.py + fas.flip_mcl.forward_eval):
  logits = image–text similarity over [spoof_ensemble, real_ensemble]
  live_prob = softmax(logits)[:, 1]   # class 1 = real/bonafide
  Argmax 1 = REAL, 0 = SPOOF

Preprocess (utils/dataset.FASDatasetSSLCLIP eval path):
  BGR→RGB, resize 224, ToTensor, ImageNet mean/std
  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  (CLIP's own preprocess is discarded during FLIP training.)

Face crop: official FAS data is pre-cropped faces; for VIGIL frames we use a
square bbox expand 1.3 (same research convention as FSFM Space crop).

License: no LICENSE file in upstream repo (verify before any commercial use).
Not connected to production authentication.
"""

from __future__ import annotations

import enum
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE_DIR = Path(__file__).resolve().parent.parent
FLIP_ROOT = Path(__file__).resolve().parent / "flip"
DEFAULT_CHECKPOINT = (
    BASE_DIR / "models" / "research" / "flip" / "msu_flip_mcl_0shot.pth.tar"
)
MODEL_NAME = "FLIP_MCL_MSU0shot"
HF_OR_SOURCE = (
    "https://github.com/koushiksrivats/FLIP/blob/main/docs/model_zoo.md "
    "(MCIO MSU 0-shot SharePoint → msu_flip_mcl.pth.tar)"
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 224
DEFAULT_BBOX_EXPAND = 1.3


class PredictedClass(str, enum.Enum):
    REAL = "REAL"
    SPOOF = "SPOOF"


@dataclass(frozen=True)
class FlipResult:
    live_prob: float
    spoof_prob: float
    predicted_class: PredictedClass
    argmax_cls: int
    inference_ms: float
    crop_mode: str
    device: str


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


class FlipResearch:
    """Isolated FLIP-MCL MSU 0-shot FAS wrapper (research only)."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        *,
        device: str | None = None,
        bbox_expand: float = DEFAULT_BBOX_EXPAND,
    ) -> None:
        if str(FLIP_ROOT) not in sys.path:
            sys.path.insert(0, str(FLIP_ROOT))

        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.bbox_expand = float(bbox_expand)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"FLIP checkpoint missing: {self.checkpoint_path}. "
                f"Download MSU 0-shot from {HF_OR_SOURCE}"
            )
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from fas import flip_mcl  # noqa: WPS433 — local upstream

        # Build on CPU first (upstream clip.load was patched to cpu), then .to(device)
        self.model = flip_mcl(in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.meta = {
            "checkpoint": str(self.checkpoint_path),
            "sha256": self.checkpoint_sha256,
            "source": HF_OR_SOURCE,
            "protocol": "MCIO_MSU_leave_out_0shot",
            "method": "flip_mcl",
            "architecture": "CLIP ViT-B/16 + multimodal contrastive (FLIP-MCL)",
            "constructor": {"in_dim": 512, "ssl_mlp_dim": 4096, "ssl_emb_dim": 256},
            "input_size": DEFAULT_INPUT_SIZE,
            "bbox_expand": self.bbox_expand,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "class_mapping": "0=SPOOF, 1=REAL (text ensembling spoof then real)",
            "score_rule": "forward_eval → softmax[:,1]=live_prob",
            "license": "No LICENSE file in upstream repo; research use only pending clarification",
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
            "device": str(self.device),
        }

    def preprocess_bgr_crop(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE), interpolation=cv2.INTER_LINEAR
        )
        x = rgb.astype(np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        x = (x - mean) / std
        x = np.transpose(x, (2, 0, 1))
        return torch.from_numpy(x).unsqueeze(0)

    @torch.inference_mode()
    def predict_crop(
        self, crop_bgr: np.ndarray, *, crop_mode: str = "bbox_expand"
    ) -> FlipResult:
        t0 = time.perf_counter()
        tensor = self.preprocess_bgr_crop(crop_bgr).to(self.device, non_blocking=True)
        # Official MCL eval path (evaluate.py comment + fas.flip_mcl.forward_eval)
        logits, _ = self.model.forward_eval(tensor, True)
        probs = F.softmax(logits.float(), dim=1)[0]
        spoof_p = float(probs[0].item())
        live_p = float(probs[1].item())
        argmax = int(torch.argmax(probs).item())
        pred = PredictedClass.REAL if argmax == 1 else PredictedClass.SPOOF
        ms = (time.perf_counter() - t0) * 1000.0
        return FlipResult(
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
    ) -> FlipResult | None:
        crop = crop_face_square_expand(frame_bgr, bbox_xyxy, expand=self.bbox_expand)
        if crop is None:
            return None
        return self.predict_crop(crop, crop_mode=f"bbox_expand_{self.bbox_expand}")

    def peak_cuda_memory_mb(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return float(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024))

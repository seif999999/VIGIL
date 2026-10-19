"""Face crop utilities for anti-spoof-mn3 (SCRFD bbox, baseline)."""

from __future__ import annotations

import cv2
import numpy as np


def crop_face_bgr(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    expand_factor: float = 1.0,
) -> np.ndarray | None:
    """
    Crop the SCRFD bounding box from a BGR frame.

    Baseline: direct bbox crop (expand_factor=1.0). Optional expand_factor scales
    width/height about the bbox center before clamping. Coordinates are clamped to
    image bounds; empty or invalid crops return None.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    if expand_factor != 1.0 and expand_factor > 0:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        bw = (x2 - x1) * expand_factor
        bh = (y2 - y1) * expand_factor
        x1 = cx - bw * 0.5
        y1 = cy - bh * 0.5
        x2 = cx + bw * 0.5
        y2 = cy + bh * 0.5
    ix1 = max(0, min(w - 1, int(round(x1))))
    iy1 = max(0, min(h - 1, int(round(y1))))
    ix2 = max(0, min(w, int(round(x2))))
    iy2 = max(0, min(h, int(round(y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    crop = frame_bgr[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return None
    return crop

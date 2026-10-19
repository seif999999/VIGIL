"""Report Webots/BGR frame dimensions and pixel stats; optional PNG save."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def analyze_bgr_frame(frame_bgr: np.ndarray) -> dict:
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "valid": False,
            "reason": "empty",
        }
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        return {
            "valid": False,
            "reason": f"unexpected_shape_{frame_bgr.shape}",
        }
    h, w = frame_bgr.shape[:2]
    flat = frame_bgr.reshape(-1, 3).astype(np.float64)
    mean_bgr = flat.mean(axis=0)
    std_bgr = flat.std(axis=0)
    return {
        "valid": True,
        "width": w,
        "height": h,
        "min": int(frame_bgr.min()),
        "max": int(frame_bgr.max()),
        "mean_bgr": [float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])],
        "std_bgr": [float(std_bgr[0]), float(std_bgr[1]), float(std_bgr[2])],
        "likely_black_feed": bool(frame_bgr.max() < 8 and frame_bgr.mean() < 3.0),
        "likely_uniform": bool(np.max(std_bgr) < 2.0),
    }


def save_diagnostic_png(frame_bgr: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"Failed to write {path}")
    return path


def print_diagnostic_report(stats: dict, path: Path | None = None) -> None:
    print("--- Webots camera frame diagnostic ---")
    if not stats.get("valid"):
        print(f"Invalid frame: {stats.get('reason', 'unknown')}")
        return
    print(f"Dimensions: {stats['width']}x{stats['height']}")
    print(f"Pixel min/max: {stats['min']} / {stats['max']}")
    print(
        f"Mean BGR: {stats['mean_bgr'][0]:.2f}, "
        f"{stats['mean_bgr'][1]:.2f}, {stats['mean_bgr'][2]:.2f}"
    )
    print(
        f"Std BGR: {stats['std_bgr'][0]:.2f}, "
        f"{stats['std_bgr'][1]:.2f}, {stats['std_bgr'][2]:.2f}"
    )
    if stats.get("likely_black_feed"):
        print("Assessment: likely BLACK or near-zero camera feed")
    elif stats.get("likely_uniform"):
        print("Assessment: nearly UNIFORM (wrong exposure or solid fill)")
    else:
        print("Assessment: non-black, non-uniform scene content present")
    if path is not None:
        print(f"Saved PNG: {path}")

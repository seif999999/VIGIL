"""Offline: SCRFD check on rendered poster texture file (not live human)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
FACE_DIR = BASE_DIR.parent / "face_detection"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check SCRFD on webots/textures/face_target.jpg",
    )
    parser.add_argument(
        "--texture",
        type=Path,
        default=BASE_DIR / "textures" / "face_target.jpg",
    )
    args = parser.parse_args()
    if not args.texture.is_file():
        print(f"[ERROR] Missing {args.texture}", file=sys.stderr)
        sys.exit(1)
    bgr = cv2.imread(str(args.texture))
    if bgr is None:
        print("[ERROR] Could not decode texture", file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, str(FACE_DIR))
    from detector import FaceDetector  # noqa: WPS433

    det = FaceDetector().detect(bgr)
    print("--- Rendered poster texture (simulation only) ---")
    print(f"File: {args.texture}")
    print(f"Size: {bgr.shape[1]}x{bgr.shape[0]}")
    print(f"SCRFD faces: {det.face_count}")
    print(f"SCRFD decision: {det.decision.value}")
    sys.exit(0 if det.face_count >= 1 else 1)


if __name__ == "__main__":
    main()

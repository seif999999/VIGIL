"""Static image: SCRFD → face-count gate → anti-spoof-mn3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from detector import MODEL_NAME, LivenessDetector
from face_bridge import load_face_detector_module
from pipeline import run_liveness_if_face_ok

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Static SCRFD + anti-spoof-mn3")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        print(f"[ERROR] Could not decode: {args.image}", file=sys.stderr)
        sys.exit(1)

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector(det_thresh=args.threshold)
    liveness = LivenessDetector()

    det = face_detector.detect(bgr)
    print(f"Face detector: {face_mod.DETECTOR_NAME}")
    print(f"Liveness model: {MODEL_NAME}")
    print(f"Faces: {det.face_count}")
    print(f"Face decision: {det.decision.value}")

    if det.decision != face_mod.FaceCountDecision.FACE_OK:
        sys.exit(0)

    live = run_liveness_if_face_ok(bgr, det, liveness)
    if live is None:
        print("[ERROR] Liveness gate failed despite FACE_OK", file=sys.stderr)
        sys.exit(1)

    print(f"Real score: {live.real_score:.6f}")
    print(f"Spoof score: {live.spoof_score:.6f}")
    print(f"Predicted class: {live.predicted_class.value}")
    print(f"Inference latency: {live.inference_ms:.2f} ms")


if __name__ == "__main__":
    main()

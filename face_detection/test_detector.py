"""Static-image SCRFD test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from detector import DETECTOR_NAME, FaceDetector, draw_detections

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Test SCRFD on one image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional annotated output path",
    )
    parser.add_argument("--show", action="store_true", help="Display window")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"[ERROR] Could not decode image: {args.image}", file=sys.stderr)
        sys.exit(1)

    detector = FaceDetector(det_thresh=args.threshold)
    result = detector.detect(image)

    print(f"Detector: {DETECTOR_NAME}")
    print(f"Model: {detector.model_path}")
    print(f"Face count: {result.face_count}")
    print(f"Decision: {result.decision.value}")
    print(f"Inference: {result.inference_ms:.2f} ms")

    for i, face in enumerate(result.faces):
        print(f"Face {i}:")
        print(f"  bbox: {face.bbox}")
        print(f"  confidence: {face.confidence:.4f}")
        print(f"  landmarks (5): {face.landmarks.tolist()}")

    annotated = draw_detections(image, result)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), annotated)
        print(f"Saved: {args.save}")

    if args.show:
        cv2.imshow("VIGIL SCRFD test", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

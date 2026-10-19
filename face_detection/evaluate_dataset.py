"""Evaluate SCRFD on a small labeled image set (CSV manifest)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

from detector import FaceDetector, face_count_decision

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE_DIR / "evaluation" / "manifest.csv"
OUTPUT_DIR = BASE_DIR / "evaluation_output"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face-count baseline")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="CSV: image_path,expected_face_count",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"[ERROR] Manifest not found: {args.manifest}", file=sys.stderr)
        print(
            "Create evaluation/manifest.csv with columns: image_path,expected_face_count",
            file=sys.stderr,
        )
        sys.exit(1)

    detector = FaceDetector(det_thresh=args.threshold)
    rows_out = []
    correct = 0
    total = 0

    with args.manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row["image_path"].strip()
            expected = int(row["expected_face_count"])
            img_path = (args.manifest.parent / rel).resolve()
            if not img_path.is_file():
                print(f"[WARN] Missing image: {img_path}")
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"[WARN] Could not decode: {img_path}")
                continue
            result = detector.detect(image)
            detected = result.face_count
            decision = face_count_decision(detected)
            ok = detected == expected
            correct += int(ok)
            total += 1
            rows_out.append(
                {
                    "image": rel,
                    "expected_face_count": expected,
                    "detected_face_count": detected,
                    "decision": decision.value,
                    "confidence_top": result.faces[0].confidence if result.faces else None,
                    "latency_ms": result.inference_ms,
                    "count_correct": ok,
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "evaluation_results.json"
    json_path.write_text(json.dumps(rows_out, indent=2), encoding="utf-8")

    print(f"Evaluated: {total}")
    print(f"Count correct: {correct}/{total}")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()

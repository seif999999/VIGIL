"""SCRFD inference microbenchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from detector import DETECTOR_NAME, FaceDetector

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "benchmark_results"
WARMUP = 50


def _stats_ms(samples_ns: list[int]) -> dict:
    ms = [x / 1_000_000 for x in samples_ns]
    ms_sorted = sorted(ms)
    p95_i = max(0, int(len(ms_sorted) * 0.95) - 1)
    return {
        "count": len(ms),
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "p95_ms": ms_sorted[p95_i],
        "min_ms": min(ms),
        "max_ms": max(ms),
        "detector_fps_from_median": 1000.0 / statistics.median(ms) if ms else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SCRFD inference")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"[ERROR] Could not decode: {args.image}", file=sys.stderr)
        sys.exit(1)

    detector = FaceDetector(det_thresh=args.threshold)

    for _ in range(WARMUP):
        detector.detect(image)

    samples: list[int] = []
    for _ in range(args.iterations):
        t0 = time.perf_counter_ns()
        result = detector.detect(image)
        t1 = time.perf_counter_ns()
        samples.append(t1 - t0)
        _ = result

    stats = _stats_ms(samples)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"detector_{ts}.json"
    payload = {
        "timestamp_utc": ts,
        "detector": DETECTOR_NAME,
        "model_path": str(detector.model_path),
        "image": str(args.image),
        "iterations": args.iterations,
        "warmup": WARMUP,
        "timing": "time.perf_counter_ns()",
        "stats": stats,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n--- VIGIL Detector Benchmark ---")
    print(f"Detector: {DETECTOR_NAME}")
    print(f"Iterations: {args.iterations}")
    print(f"Mean:   {stats['mean_ms']:.4f} ms")
    print(f"Median: {stats['median_ms']:.4f} ms")
    print(f"P95:    {stats['p95_ms']:.4f} ms")
    print(f"Min:    {stats['min_ms']:.4f} ms")
    print(f"Max:    {stats['max_ms']:.4f} ms")
    print(f"Detector FPS (from median): {stats['detector_fps_from_median']:.1f}")
    print(f"JSON: {out_path}")


if __name__ == "__main__":
    main()

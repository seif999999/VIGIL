"""Benchmark anti-spoof-mn3 (liveness-only and optional SCRFD+liveness)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from detector import MODEL_NAME, LivenessDetector
from face_bridge import load_face_detector_module

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "benchmark_results"
WARMUP = 50
INPUT_W = 128
INPUT_H = 128


def _stats_ms(samples_ns: list[int]) -> dict:
    ms = [x / 1_000_000 for x in samples_ns]
    ms_sorted = sorted(ms)
    p95_i = max(0, int(len(ms_sorted) * 0.95) - 1)
    med = statistics.median(ms) if ms else 0.0
    return {
        "count": len(ms),
        "mean_ms": statistics.mean(ms) if ms else 0.0,
        "median_ms": med,
        "p95_ms": ms_sorted[p95_i] if ms_sorted else 0.0,
        "min_ms": min(ms) if ms else 0.0,
        "max_ms": max(ms) if ms else 0.0,
        "liveness_fps_from_median": 1000.0 / med if med > 0 else 0.0,
    }


def _preloaded_crop(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"Could not decode {image_path}")
    return cv2.resize(bgr, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark anti-spoof-mn3")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"[ERROR] Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    liveness = LivenessDetector()
    crop = _preloaded_crop(args.image)
    blob = LivenessDetector.preprocess_face_crop_bgr(crop)
    input_name = liveness._session.get_inputs()[0].name  # noqa: SLF001
    output_name = liveness._session.get_outputs()[0].name  # noqa: SLF001
    session = liveness._session  # noqa: SLF001

    for _ in range(WARMUP):
        session.run([output_name], {input_name: blob})

    samples: list[int] = []
    for _ in range(args.iterations):
        t0 = time.perf_counter_ns()
        session.run([output_name], {input_name: blob})
        samples.append(time.perf_counter_ns() - t0)

    liveness_stats = _stats_ms(samples)

    combined_stats = None
    face_mod = load_face_detector_module()
    full_bgr = cv2.imread(str(args.image))
    if full_bgr is not None:
        face_detector = face_mod.FaceDetector(det_thresh=args.threshold)
        for _ in range(WARMUP):
            det = face_detector.detect(full_bgr)
            if det.face_count == 1:
                crop_f = cv2.resize(
                    full_bgr[
                        int(det.faces[0].y1) : int(det.faces[0].y2),
                        int(det.faces[0].x1) : int(det.faces[0].x2),
                    ],
                    (INPUT_W, INPUT_H),
                )
                liveness.predict(crop_f)

        comb: list[int] = []
        for _ in range(min(args.iterations, 200)):
            t0 = time.perf_counter_ns()
            det = face_detector.detect(full_bgr)
            if det.face_count == 1:
                from pipeline import run_liveness_if_face_ok

                run_liveness_if_face_ok(full_bgr, det, liveness)
            comb.append(time.perf_counter_ns() - t0)
        if comb:
            combined_stats = _stats_ms(comb)
            combined_stats["note"] = (
                "SCRFD + liveness when FACE_OK; capped at 200 iters if face missing"
            )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"liveness_{ts}.json"
    payload = {
        "timestamp_utc": ts,
        "model": MODEL_NAME,
        "model_path": str(liveness.model_path),
        "image": str(args.image),
        "iterations": args.iterations,
        "warmup": WARMUP,
        "timing": "time.perf_counter_ns(); liveness-only excludes SCRFD",
        "liveness_only": liveness_stats,
        "scrfd_plus_liveness": combined_stats,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n--- VIGIL Liveness Benchmark ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Liveness-only iterations: {args.iterations}")
    print(f"Mean:   {liveness_stats['mean_ms']:.4f} ms")
    print(f"Median: {liveness_stats['median_ms']:.4f} ms")
    print(f"P95:    {liveness_stats['p95_ms']:.4f} ms")
    print(f"Liveness FPS (median): {liveness_stats['liveness_fps_from_median']:.1f}")
    if combined_stats:
        print(f"SCRFD+Liveness median: {combined_stats['median_ms']:.4f} ms")
    print(f"JSON: {out_path}")


if __name__ == "__main__":
    main()

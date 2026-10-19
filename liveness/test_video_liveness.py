"""Offline video: SCRFD → FACE_OK → 1.0x crop → anti-spoof-mn3 (baseline pipeline)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from decision_policy import (
    PROVISIONAL_POLICY_NOTE,
    ProvisionalDecisionPolicy,
    ProvisionalDemoOutcome,
    resolve_demo_outcome,
)
from detector import LivenessDetector, MODEL_NAME
from face_bridge import load_face_detector_module
from pipeline import run_liveness_if_face_ok

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "video_results"

LAPTOP_WEBCAM_REAL_RANGE = (0.004, 0.007)
LAPTOP_WEBCAM_SPOOF_RANGE = (0.993, 0.996)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline video liveness baseline")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--provisional-real-min",
        type=float,
        default=0.85,
        help="Provisional min real score for confident REAL demo outcome (uncalibrated)",
    )
    parser.add_argument(
        "--provisional-spoof-min",
        type=float,
        default=0.85,
        help="Provisional min spoof score for confident SPOOF demo outcome (uncalibrated)",
    )
    args = parser.parse_args()

    policy = ProvisionalDecisionPolicy(
        confident_real_min=args.provisional_real_min,
        confident_spoof_min=args.provisional_spoof_min,
    )

    if not args.video.is_file():
        print(f"[ERROR] Video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = (
        frame_count_prop / orig_fps
        if orig_fps and orig_fps > 0 and frame_count_prop > 0
        else None
    )

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector(det_thresh=args.threshold)
    liveness = LivenessDetector()

    total = 0
    no_face = 0
    multiple = 0
    face_ok = 0
    liveness_runs = 0
    real_scores: list[float] = []
    spoof_scores: list[float] = []
    pred_real = 0
    pred_spoof = 0
    liveness_latencies: list[float] = []
    combined_latencies: list[float] = []
    demo_counts = {o.value: 0 for o in ProvisionalDemoOutcome}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        total += 1

        t0 = time.perf_counter_ns()
        det = face_detector.detect(frame)
        decision = det.decision

        decision_value = decision.value

        if decision == face_mod.FaceCountDecision.NO_FACE:
            no_face += 1
            demo_counts[
                resolve_demo_outcome(decision_value, None, policy).value
            ] += 1
            continue
        if decision == face_mod.FaceCountDecision.MULTIPLE_FACES:
            multiple += 1
            demo_counts[
                resolve_demo_outcome(decision_value, None, policy).value
            ] += 1
            continue

        face_ok += 1
        live = run_liveness_if_face_ok(frame, det, liveness)
        combined_ms = (time.perf_counter_ns() - t0) / 1_000_000
        combined_latencies.append(combined_ms)

        demo_counts[
            resolve_demo_outcome(decision_value, live, policy).value
        ] += 1

        if live is None:
            continue

        liveness_runs += 1
        real_scores.append(live.real_score)
        spoof_scores.append(live.spoof_score)
        liveness_latencies.append(live.inference_ms)
        if live.predicted_class.value == "REAL":
            pred_real += 1
        else:
            pred_spoof += 1

    cap.release()

    if duration_s is None and total > 0 and orig_fps and orig_fps > 0:
        duration_s = total / orig_fps

    avg_real = statistics.mean(real_scores) if real_scores else 0.0
    avg_spoof = statistics.mean(spoof_scores) if spoof_scores else 0.0
    med_real = statistics.median(real_scores) if real_scores else 0.0
    med_spoof = statistics.median(spoof_scores) if spoof_scores else 0.0
    min_real = min(real_scores) if real_scores else 0.0
    max_real = max(real_scores) if real_scores else 0.0
    med_live_ms = statistics.median(liveness_latencies) if liveness_latencies else 0.0
    med_combined_ms = (
        statistics.median(combined_latencies) if combined_latencies else 0.0
    )
    pct_real = (100.0 * pred_real / liveness_runs) if liveness_runs else 0.0
    pct_spoof = (100.0 * pred_spoof / liveness_runs) if liveness_runs else 0.0
    overall = "REAL" if avg_real > avg_spoof else "SPOOF"

    def frame_rate_pct(count: int) -> float:
        return (100.0 * count / total) if total else 0.0

    demo_rates_pct = {
        key: frame_rate_pct(demo_counts[key]) for key in demo_counts
    }

    print("\n--- VIGIL Offline Video Liveness Summary ---")
    print(f"Video: {args.video}")
    print(f"Duration: {duration_s:.2f} s" if duration_s else "Duration: (unknown)")
    print(f"Original resolution: {orig_w}x{orig_h}")
    print(f"Original FPS: {orig_fps:.3f}" if orig_fps and orig_fps > 0 else "Original FPS: (unknown)")
    print(f"Total decoded frames: {total}")
    print(f"FACE_OK frames: {face_ok}")
    print(f"NO_FACE frames: {no_face}")
    print(f"MULTIPLE_FACES frames: {multiple}")
    print(f"Liveness runs: {liveness_runs}")
    print(f"Liveness model: {MODEL_NAME}")
    print()
    print(f"Average REAL score: {avg_real:.6f}")
    print(f"Average SPOOF score: {avg_spoof:.6f}")
    print(f"Median REAL score: {med_real:.6f}")
    print(f"Median SPOOF score: {med_spoof:.6f}")
    print(f"Minimum REAL score: {min_real:.6f}")
    print(f"Maximum REAL score: {max_real:.6f}")
    print(f"Frames predicted REAL: {pred_real}")
    print(f"Frames predicted SPOOF: {pred_spoof}")
    print(f"Percentage predicted REAL: {pct_real:.2f}%")
    print(f"Percentage predicted SPOOF: {pct_spoof:.2f}%")
    print(f"Overall class (from mean scores): {overall}")
    print(f"Median liveness latency: {med_live_ms:.3f} ms")
    print(f"Median SCRFD+liveness latency: {med_combined_ms:.3f} ms")

    print("\n--- Provisional demo policy (uncalibrated thresholds) ---")
    print(PROVISIONAL_POLICY_NOTE)
    print(
        f"Thresholds: REAL>={policy.confident_real_min}, "
        f"SPOOF>={policy.confident_spoof_min}; else UNCERTAIN_REVIEW"
    )
    for outcome in ProvisionalDemoOutcome:
        key = outcome.value
        print(
            f"  {key}: {demo_counts[key]} frames "
            f"({demo_rates_pct[key]:.2f}% of decoded frames)"
        )

    print("\n--- Comparison note (raw model output only) ---")
    print(
        "Laptop webcam live baseline (prior): REAL approximately "
        f"{LAPTOP_WEBCAM_REAL_RANGE[0]:.3f}–{LAPTOP_WEBCAM_REAL_RANGE[1]:.3f}, "
        f"SPOOF approximately "
        f"{LAPTOP_WEBCAM_SPOOF_RANGE[0]:.3f}–{LAPTOP_WEBCAM_SPOOF_RANGE[1]:.3f}"
    )
    print(f"Phone video (this run): avg REAL {avg_real:.6f}, avg SPOOF {avg_spoof:.6f}")
    substantially_different = avg_real > 0.05 or avg_spoof < 0.95
    if liveness_runs == 0:
        print("Phone video: no liveness runs — cannot compare scores.")
    elif substantially_different:
        print(
            "Phone video scores are substantially different from the laptop-webcam baseline "
            "(by simple magnitude check on averages)."
        )
    else:
        print(
            "Phone video averages are in a similar SPOOF-dominant range as the laptop baseline."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp_utc": ts,
        "video_path": str(args.video),
        "duration_s": duration_s,
        "original_resolution": {"width": orig_w, "height": orig_h},
        "original_fps": orig_fps,
        "total_decoded_frames": total,
        "face_ok_frames": face_ok,
        "no_face_frames": no_face,
        "multiple_faces_frames": multiple,
        "liveness_runs": liveness_runs,
        "average_real": avg_real,
        "average_spoof": avg_spoof,
        "median_real": med_real,
        "median_spoof": med_spoof,
        "min_real": min_real,
        "max_real": max_real,
        "frames_predicted_real": pred_real,
        "frames_predicted_spoof": pred_spoof,
        "pct_predicted_real": pct_real,
        "pct_predicted_spoof": pct_spoof,
        "overall_from_means": overall,
        "median_liveness_ms": med_live_ms,
        "median_scrfd_plus_liveness_ms": med_combined_ms,
        "provisional_demo_policy": {
            "note": PROVISIONAL_POLICY_NOTE,
            "thresholds_uncalibrated": {
                "confident_real_min": policy.confident_real_min,
                "confident_spoof_min": policy.confident_spoof_min,
            },
            "frame_counts": demo_counts,
            "frame_rates_pct_of_decoded": demo_rates_pct,
        },
    }
    out_path = RESULTS_DIR / f"video_{ts}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON summary: {out_path}")


if __name__ == "__main__":
    main()

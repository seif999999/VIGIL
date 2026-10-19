"""Live webcam: SCRFD → FACE_OK gate → anti-spoof-mn3."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from detector import MODEL_NAME, LivenessDetector
from face_bridge import load_face_detector_module
from decision_policy import ProvisionalDecisionPolicy, resolve_demo_outcome
from overlay import draw_demo_liveness_readout
from pipeline import run_liveness_if_face_ok
from webcam_capture import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    open_webcam,
    print_resolution_info,
)

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "manual_logs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SCRFD + anti-spoof-mn3")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--scenario",
        choices=("REAL", "PRINT", "SCREEN"),
        default=None,
        help="Optional manual attack label for CSV logging",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=None,
        help="Append one summary row per run (scores are run averages)",
    )
    parser.add_argument(
        "--dshow-device",
        type=str,
        default=None,
        help='Windows DirectShow device via FFmpeg, e.g. "Nazmy (Windows Virtual Camera)"',
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Requested FPS for FFmpeg DirectShow capture",
    )
    parser.add_argument(
        "--stream-url",
        type=str,
        default=None,
        help="Live HTTP/MJPEG or RTSP stream URL from phone/app",
    )
    parser.add_argument(
        "--windows-device",
        type=str,
        default=None,
        help='Native Windows MediaCapture device, e.g. "Nazmy (Windows Virtual Camera)"',
    )
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

    from capture_sources import validate_exclusive_live_sources

    validate_exclusive_live_sources(
        windows_device=args.windows_device,
        stream_url=args.stream_url,
        dshow_device=args.dshow_device,
    )

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector(det_thresh=args.threshold)
    liveness = LivenessDetector()
    cap = None
    ok_probe = False
    probe_frame = None
    source_label = "OPENCV"
    if args.windows_device:
        from windows_camera_capture import (  # noqa: WPS433
            open_winrt_capture,
            print_winrt_capture_info,
        )

        try:
            cap = open_winrt_capture(args.windows_device)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print_winrt_capture_info(cap.capture_info)
        ok_probe, probe_frame = cap.read()
        source_label = "WINDOWS"
    elif args.stream_url:
        from network_capture import (  # noqa: WPS433
            open_network_stream,
            print_network_capture_info,
        )

        try:
            cap = open_network_stream(args.stream_url)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print_network_capture_info(cap.capture_info)
        ok_probe, probe_frame = cap.read()
        source_label = "NETWORK"
    elif args.dshow_device:
        from ffmpeg_capture import (  # noqa: WPS433
            open_dshow_capture,
            print_dshow_capture_info,
        )

        try:
            cap = open_dshow_capture(
                args.dshow_device,
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        ok_probe, probe_frame = cap.read()
        actual_wh = None
        if ok_probe and probe_frame is not None:
            h, w = probe_frame.shape[:2]
            actual_wh = (w, h)
        print_dshow_capture_info(cap.capture_info, actual_wh=actual_wh)
        source_label = "DSHOW"
    else:
        try:
            cap, cam_info = open_webcam(
                args.camera, width=args.width, height=args.height
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print_resolution_info(cam_info)

    window = "VIGIL - SCRFD + Liveness Live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frames = 0
    liveness_runs = 0
    real_scores: list[float] = []
    spoof_scores: list[float] = []
    latencies: list[float] = []
    combined_latencies: list[float] = []
    demo_outcome_counts: dict[str, int] = {}
    started = time.perf_counter()

    pending_frame = None
    if ok_probe and probe_frame is not None:
        pending_frame = probe_frame

    try:
        while True:
            if args.duration is not None and time.perf_counter() - started >= args.duration:
                break
            cv2.pollKey()
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                ok = True
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                continue

            t0 = time.perf_counter_ns()
            det = face_detector.detect(frame)
            live = run_liveness_if_face_ok(frame, det, liveness)
            total_ms = (time.perf_counter_ns() - t0) / 1_000_000

            frames += 1
            combined_latencies.append(total_ms)
            display = face_mod.draw_detections(frame, det)
            demo_label = resolve_demo_outcome(
                det.decision.value, live, policy
            ).value
            demo_outcome_counts[demo_label] = (
                demo_outcome_counts.get(demo_label, 0) + 1
            )
            if live is not None:
                liveness_runs += 1
                real_scores.append(live.real_score)
                spoof_scores.append(live.spoof_score)
                latencies.append(live.inference_ms)
            display = draw_demo_liveness_readout(
                display,
                face_decision_value=det.decision.value,
                liveness=live,
                policy=policy,
                total_ms=total_ms if live is not None else None,
            )

            cv2.putText(
                display,
                f"Source: {source_label}",
                (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                f"Liveness model: {MODEL_NAME}",
                (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    med_live = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    med_comb = (
        sorted(combined_latencies)[len(combined_latencies) // 2]
        if combined_latencies
        else 0.0
    )
    avg_real = sum(real_scores) / len(real_scores) if real_scores else 0.0
    avg_spoof = sum(spoof_scores) / len(spoof_scores) if spoof_scores else 0.0
    pred = "N/A"
    if real_scores and spoof_scores:
        pred = "REAL" if avg_real > avg_spoof else "SPOOF"

    print("\n--- VIGIL Live Liveness Summary ---")
    print(f"Source: {source_label}")
    print(f"Frames processed: {frames}")
    print(f"Liveness runs (FACE_OK): {liveness_runs}")
    print(f"Avg real score: {avg_real:.4f}")
    print(f"Avg spoof score: {avg_spoof:.4f}")
    print(f"Avg predicted class (from means): {pred}")
    print(f"Median liveness latency: {med_live:.2f} ms")
    print(f"Median SCRFD+liveness latency: {med_comb:.2f} ms")
    print(
        f"Provisional policy: real>={policy.confident_real_min}, "
        f"spoof>={policy.confident_spoof_min} (uncalibrated — demo only)"
    )
    if demo_outcome_counts:
        print("Provisional demo outcome frame counts:")
        for key in sorted(demo_outcome_counts):
            print(f"  {key}: {demo_outcome_counts[key]}")

    if args.log_csv is not None or args.scenario is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = args.log_csv or LOG_DIR / "manual_runs.csv"
        write_header = not path.is_file()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "timestamp_utc",
                        "scenario",
                        "frames",
                        "liveness_runs",
                        "avg_real_score",
                        "avg_spoof_score",
                        "predicted_class",
                        "median_liveness_ms",
                        "median_combined_ms",
                    ]
                )
            w.writerow(
                [
                    datetime.now(timezone.utc).isoformat(),
                    args.scenario or "",
                    frames,
                    liveness_runs,
                    f"{avg_real:.6f}",
                    f"{avg_spoof:.6f}",
                    pred,
                    f"{med_live:.3f}",
                    f"{med_comb:.3f}",
                ]
            )
        print(f"Log appended: {path}")


if __name__ == "__main__":
    main()

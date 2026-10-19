"""Live webcam SCRFD validation (no HMAC)."""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from detector import DETECTOR_NAME, FaceDetector, FaceCountDecision, draw_detections


def _open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {index}.", file=sys.stderr)
        raise SystemExit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main() -> None:
    parser = argparse.ArgumentParser(description="Live SCRFD face detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    detector = FaceDetector(det_thresh=args.threshold)
    cap = _open_camera(args.camera)
    window = "VIGIL - SCRFD Live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frames = 0
    face_ok = 0
    latencies: list[float] = []
    started = time.perf_counter()

    try:
        while True:
            if args.duration is not None and time.perf_counter() - started >= args.duration:
                break
            cv2.pollKey()
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            result = detector.detect(frame)
            frames += 1
            latencies.append(result.inference_ms)
            if result.decision == FaceCountDecision.FACE_OK:
                face_ok += 1

            display = draw_detections(frame, result)
            if latencies:
                fps = 1000.0 / result.inference_ms if result.inference_ms > 0 else 0.0
                cv2.putText(
                    display,
                    f"Detector FPS: {fps:.1f}",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        med = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
        print("\n--- VIGIL Live Detector Summary ---")
        print(f"Detector: {DETECTOR_NAME}")
        print(f"Frames processed: {frames}")
        print(f"Frames with exactly one face: {face_ok}")
        print(f"Median detection latency: {med:.2f} ms")


if __name__ == "__main__":
    main()

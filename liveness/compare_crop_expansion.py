"""Controlled 1.0x vs 1.2x SCRFD bbox crop comparison for anti-spoof-mn3."""

from __future__ import annotations

import argparse
import sys
import time

import cv2

from crop import crop_face_bgr
from detector import LivenessDetector, MODEL_NAME
from face_bridge import load_face_detector_module
from webcam_capture import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    open_webcam,
    print_resolution_info,
)

CROP_A = 1.0
CROP_B = 1.2


def _crop_size(crop) -> tuple[int, int]:
    if crop is None:
        return 0, 0
    h, w = crop.shape[:2]
    return w, h


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 1.0x vs 1.2x face crop for liveness (same frame)",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector(det_thresh=args.threshold)
    liveness = LivenessDetector()

    try:
        cap, cam_info = open_webcam(
            args.camera, width=args.width, height=args.height
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print_resolution_info(cam_info)

    window = "VIGIL - Crop 1.0x vs 1.2x"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    n_compared = 0
    sum_real_a = sum_spoof_a = 0.0
    sum_real_b = sum_spoof_b = 0.0
    started = time.perf_counter()

    print(
        f"\n[INFO] Comparing crops on FACE_OK frames for {args.duration}s. "
        "Press Q to stop early.\n"
    )

    try:
        while time.perf_counter() - started < args.duration:
            cv2.pollKey()
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            det = face_detector.detect(frame)
            if det.decision != face_mod.FaceCountDecision.FACE_OK:
                display = face_mod.draw_detections(frame, det)
                cv2.imshow(window, display)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    break
                continue

            bbox = det.faces[0].bbox
            crop_a = crop_face_bgr(frame, bbox, expand_factor=CROP_A)
            crop_b = crop_face_bgr(frame, bbox, expand_factor=CROP_B)
            if crop_a is None or crop_b is None:
                continue

            res_a = liveness.predict(crop_a)
            res_b = liveness.predict(crop_b)
            n_compared += 1
            sum_real_a += res_a.real_score
            sum_spoof_a += res_a.spoof_score
            sum_real_b += res_b.real_score
            sum_spoof_b += res_b.spoof_score

            wa, ha = _crop_size(crop_a)
            wb, hb = _crop_size(crop_b)

            print(
                f"Frame {n_compared} | "
                f"1.0x {wa}x{ha} REAL={res_a.real_score:.4f} SPOOF={res_a.spoof_score:.4f} "
                f"{res_a.predicted_class.value} | "
                f"1.2x {wb}x{hb} REAL={res_b.real_score:.4f} SPOOF={res_b.spoof_score:.4f} "
                f"{res_b.predicted_class.value}"
            )

            display = face_mod.draw_detections(frame, det)
            y = 120
            for line in (
                f"Crop compare ({MODEL_NAME})",
                f"1.0x {wa}x{ha} R={res_a.real_score:.3f} S={res_a.spoof_score:.3f} "
                f"{res_a.predicted_class.value}",
                f"1.2x {wb}x{hb} R={res_b.real_score:.3f} S={res_b.spoof_score:.3f} "
                f"{res_b.predicted_class.value}",
            ):
                cv2.putText(
                    display,
                    line,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 200, 0),
                    1,
                    cv2.LINE_AA,
                )
                y += 20
            cv2.imshow(window, display)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\n--- VIGIL Crop Expansion Comparison Summary ---")
    print(f"Compared frames (FACE_OK): {n_compared}")
    if n_compared > 0:
        print(f"1.0x average REAL:  {sum_real_a / n_compared:.6f}")
        print(f"1.0x average SPOOF: {sum_spoof_a / n_compared:.6f}")
        print(f"1.2x average REAL:  {sum_real_b / n_compared:.6f}")
        print(f"1.2x average SPOOF: {sum_spoof_b / n_compared:.6f}")
    else:
        print("[WARN] No FACE_OK frames compared.")


if __name__ == "__main__":
    main()

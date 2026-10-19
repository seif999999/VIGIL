"""One-frame diagnostic: live liveness path vs offline reload of the same face crop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from crop import crop_face_bgr
from detector import INPUT_SIZE, LivenessDetector, MODEL_NAME
from face_bridge import load_face_detector_module
from pipeline import run_liveness_if_face_ok
from webcam_capture import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    open_webcam,
    print_resolution_info,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BASE_DIR / "diagnostic_output"


def _result_dict(result) -> dict:
    return {
        "real_score": result.real_score,
        "spoof_score": result.spoof_score,
        "predicted_class": result.predicted_class.value,
        "inference_ms": result.inference_ms,
    }


def _tensor_stats(blob: np.ndarray) -> dict:
    return {
        "shape": list(blob.shape),
        "dtype": str(blob.dtype),
        "min": float(np.min(blob)),
        "max": float(np.max(blob)),
        "mean": float(np.mean(blob)),
    }


def _score_delta(a: dict, b: dict) -> dict:
    return {
        "real_score_abs_diff": abs(a["real_score"] - b["real_score"]),
        "spoof_score_abs_diff": abs(a["spoof_score"] - b["spoof_score"]),
        "prediction_match": a["predicted_class"] == b["predicted_class"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare live vs offline liveness on one FACE_OK webcam frame",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for exactly one face (default 60)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Directory for saved frames and report JSON",
    )
    parser.add_argument(
        "--dshow-device",
        type=str,
        default=None,
        help='FFmpeg DirectShow device, e.g. "Nazmy (Windows Virtual Camera)"',
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--stream-url",
        type=str,
        default=None,
        help="Live HTTP/MJPEG or RTSP stream URL",
    )
    parser.add_argument(
        "--windows-device",
        type=str,
        default=None,
        help='Native Windows MediaCapture device name',
    )
    args = parser.parse_args()

    from capture_sources import validate_exclusive_live_sources

    validate_exclusive_live_sources(
        windows_device=args.windows_device,
        stream_url=args.stream_url,
        dshow_device=args.dshow_device,
    )

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector(det_thresh=args.threshold)
    liveness = LivenessDetector()
    cam_info = None
    dshow_info = None
    network_info = None
    winrt_info = None
    if args.windows_device:
        from windows_camera_capture import open_winrt_capture, print_winrt_capture_info

        try:
            cap = open_winrt_capture(args.windows_device)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        winrt_info = cap.capture_info
        print_winrt_capture_info(winrt_info)
        ok0, f0 = cap.read()
    elif args.stream_url:
        from network_capture import open_network_stream, print_network_capture_info

        try:
            cap = open_network_stream(args.stream_url)
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        network_info = cap.capture_info
        print_network_capture_info(network_info)
        ok0, f0 = cap.read()
    elif args.dshow_device:
        from ffmpeg_capture import open_dshow_capture, print_dshow_capture_info

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
        dshow_info = cap.capture_info
        ok0, f0 = cap.read()
        aw = (f0.shape[1], f0.shape[0]) if ok0 and f0 is not None else None
        print_dshow_capture_info(dshow_info, actual_wh=aw)
    else:
        try:
            cap, cam_info = open_webcam(
                args.camera, width=args.width, height=args.height
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        print_resolution_info(cam_info)
        ok0, f0 = False, None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Waiting for exactly one face (FACE_OK). Press Q to abort.")
    frame_bgr = None
    det = None
    deadline = time.perf_counter() + args.timeout

    pending = f0 if ok0 and f0 is not None else None

    try:
        while time.perf_counter() < deadline:
            cv2.pollKey()
            if pending is not None:
                frame = pending
                pending = None
                ok = True
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                continue
            det = face_detector.detect(frame)
            if det.decision == face_mod.FaceCountDecision.FACE_OK:
                frame_bgr = frame
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("[INFO] Aborted by user.")
                sys.exit(0)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if frame_bgr is None or det is None:
        print(
            f"[ERROR] No FACE_OK frame within {args.timeout}s.",
            file=sys.stderr,
        )
        sys.exit(1)

    face = det.faces[0]
    bbox = face.bbox
    crop_bgr = crop_face_bgr(frame_bgr, bbox)
    if crop_bgr is None:
        print("[ERROR] Face crop failed.", file=sys.stderr)
        sys.exit(1)

    fh, fw = frame_bgr.shape[:2]
    ch, cw = crop_bgr.shape[:2]
    crop_128 = cv2.resize(crop_bgr, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)

    original_path = out_dir / "original_webcam_frame.png"
    crop_path = out_dir / "face_crop_before_preprocess.png"
    resized_path = out_dir / "face_crop_128x128_visualization.png"
    cv2.imwrite(str(original_path), frame_bgr)
    cv2.imwrite(str(crop_path), crop_bgr)
    cv2.imwrite(str(resized_path), crop_128)

    # Live pipeline path (same as live_liveness.py)
    live_pipeline_result = run_liveness_if_face_ok(frame_bgr, det, liveness)
    # Direct predict on in-memory crop (same as predict_from_frame_bbox internals)
    live_direct_result = liveness.predict(crop_bgr)

    # Offline: reload saved crop file (static file path)
    crop_reloaded = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop_reloaded is None:
        print("[ERROR] Could not reload saved crop.", file=sys.stderr)
        sys.exit(1)
    offline_result = liveness.predict(crop_reloaded)

    blob_live = LivenessDetector.preprocess_face_crop_bgr(crop_bgr)
    blob_reloaded = LivenessDetector.preprocess_face_crop_bgr(crop_reloaded)
    blob_max_abs_diff = float(np.max(np.abs(blob_live - blob_reloaded)))

    live_pipe = _result_dict(live_pipeline_result)
    live_direct = _result_dict(live_direct_result)
    offline = _result_dict(offline_result)

    capture_meta: dict = {}
    if winrt_info is not None:
        capture_meta = {
            "capture_method": "windows_mediacapture",
            "windows_device": winrt_info.device_name,
            "device_id": winrt_info.device_id,
            "actual_frame": {
                "width": winrt_info.actual_width,
                "height": winrt_info.actual_height,
            },
            "pixel_format": winrt_info.pixel_format,
        }
    elif network_info is not None:
        capture_meta = {
            "capture_method": "network_stream",
            "stream_url": network_info.stream_url,
            "actual_frame": {
                "width": network_info.actual_width,
                "height": network_info.actual_height,
            },
            "reported_fps": network_info.reported_fps,
        }
    elif dshow_info is not None:
        capture_meta = {
            "capture_method": "ffmpeg_dshow",
            "dshow_device": dshow_info.device_name,
            "requested_resolution": {
                "width": dshow_info.requested_width,
                "height": dshow_info.requested_height,
            },
            "requested_fps": dshow_info.requested_fps,
            "ffmpeg_command": dshow_info.ffmpeg_command,
        }
    elif cam_info is not None:
        capture_meta = {
            "capture_method": "opencv_index",
            "webcam_capture_requested": {
                "width": cam_info.requested_width,
                "height": cam_info.requested_height,
            },
            "webcam_capture_actual": {
                "width": cam_info.actual_width,
                "height": cam_info.actual_height,
                "used_fallback_640x480": cam_info.used_fallback,
            },
        }

    report = {
        "timestamp_utc": ts,
        "model": MODEL_NAME,
        **capture_meta,
        "webcam_frame_resolution": {"width": fw, "height": fh},
        "scrfd_bbox_xyxy": {
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
        },
        "scrfd_detection_confidence": float(face.confidence),
        "crop_before_resize": {"width": cw, "height": ch},
        "crop_resize_target": {"width": INPUT_SIZE[0], "height": INPUT_SIZE[1]},
        "preprocessing_notes": {
            "single_preprocess_function": "LivenessDetector.preprocess_face_crop_bgr",
            "rgb_conversion": "once inside preprocess (BGR crop -> resize -> BGR2RGB)",
            "normalization": "once inside preprocess ((rgb - OMZ mean) / OMZ scale)",
            "no_alternate_live_pipeline": True,
        },
        "input_tensor_live_crop": _tensor_stats(blob_live),
        "input_tensor_reloaded_crop": _tensor_stats(blob_reloaded),
        "input_tensor_max_abs_diff_live_vs_reloaded": blob_max_abs_diff,
        "saved_files": {
            "original_webcam_frame": str(original_path),
            "face_crop_before_preprocess": str(crop_path),
            "face_crop_128x128_visualization": str(resized_path),
        },
        "scores": {
            "live_path_run_liveness_if_face_ok": live_pipe,
            "live_path_predict_in_memory_crop": live_direct,
            "offline_path_predict_saved_crop_file": offline,
        },
        "comparison": {
            "live_pipeline_vs_live_direct": _score_delta(live_pipe, live_direct),
            "live_pipeline_vs_offline_saved_crop": _score_delta(live_pipe, offline),
            "live_direct_vs_offline_saved_crop": _score_delta(live_direct, offline),
        },
        "interpretation_hint": (
            "If offline saved-crop scores match live scores (small diff), "
            "preprocessing is consistent; large live/static diff on the same "
            "saved file would indicate an implementation mismatch."
        ),
    }

    report_path = out_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- VIGIL Liveness Diagnostic (one FACE_OK frame) ---")
    print(f"Output directory: {out_dir}")
    print(f"Webcam resolution: {fw}x{fh}")
    print(f"SCRFD bbox: {bbox}")
    print(f"Crop size (before resize): {cw}x{ch}")
    print(f"Tensor shape: {blob_live.shape}, dtype: {blob_live.dtype}")
    print(
        f"Tensor stats (live crop): min={report['input_tensor_live_crop']['min']:.6f} "
        f"max={report['input_tensor_live_crop']['max']:.6f} "
        f"mean={report['input_tensor_live_crop']['mean']:.6f}"
    )
    print(f"Tensor max |live - reloaded|: {blob_max_abs_diff:.8f}")
    print("\nLIVE PATH (run_liveness_if_face_ok):")
    print(f"  real={live_pipe['real_score']:.6f} spoof={live_pipe['spoof_score']:.6f} "
          f"pred={live_pipe['predicted_class']}")
    print("LIVE PATH (predict in-memory crop):")
    print(f"  real={live_direct['real_score']:.6f} spoof={live_direct['spoof_score']:.6f} "
          f"pred={live_direct['predicted_class']}")
    print("OFFLINE (cv2.imread saved crop + predict):")
    print(f"  real={offline['real_score']:.6f} spoof={offline['spoof_score']:.6f} "
          f"pred={offline['predicted_class']}")
    d = report["comparison"]["live_pipeline_vs_offline_saved_crop"]
    print("\nDelta live pipeline vs offline saved crop:")
    print(f"  |d real|={d['real_score_abs_diff']:.8f} |d spoof|={d['spoof_score_abs_diff']:.8f} "
          f"same_pred={d['prediction_match']}")
    print(f"\nFull report: {report_path}")


if __name__ == "__main__":
    main()

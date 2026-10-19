"""Diagnosis-only audit: anti-spoof-mn3 integration, crops, camera domain (no prod changes)."""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from crop import crop_face_bgr
from detector import (
    INPUT_SIZE,
    LivenessDetector,
    MODEL_NAME,
    _MEAN_RGB,
    _SCALE_RGB,
)
from diagnose_offline_crop_expansion import classify_pose_bucket, resolve_run_path
from face_bridge import load_face_detector_module

BASE_DIR = Path(__file__).resolve().parent
HEAD_RUN = BASE_DIR / "diagnostic_head_motion" / "20261015T102037Z"
PHONE_VIDEO = BASE_DIR.parent / "face_detection" / "test video.mp4"
NAZMY_DIAG = BASE_DIR / "diagnostic_output" / "20261007T145351Z" / "diagnostic_report.json"


def omz_reference_preprocess_bgr(crop_bgr: np.ndarray) -> np.ndarray:
    """OMZ Original Model README: RGB, mean/scale, 128x128 linear resize, NCHW."""
    resized = cv2.resize(crop_bgr, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = (rgb - _MEAN_RGB) / _SCALE_RGB
    return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)


def run_onnx(session, blob: np.ndarray) -> tuple[float, float]:
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    (raw,) = session.run([out], {inp: blob})
    scores = np.asarray(raw, dtype=np.float32).reshape(-1)
    if scores.shape[0] != 2:
        raise RuntimeError(scores.shape)
    r, s = float(scores[0]), float(scores[1])
    if r + s > 1.01 or r + s < 0.99:
        exp = np.exp(scores - np.max(scores))
        prob = exp / np.sum(exp)
        r, s = float(prob[0]), float(prob[1])
    return r, s


def variant_preprocess(crop_bgr: np.ndarray, variant: str) -> np.ndarray:
    resized = cv2.resize(
        crop_bgr,
        INPUT_SIZE,
        interpolation=cv2.INTER_LINEAR
        if variant not in ("interp_area",)
        else cv2.INTER_AREA,
    )
    if variant in ("omz_rgb", "interp_area"):
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - _MEAN_RGB) / _SCALE_RGB
    elif variant == "bgr_converted_ir_style":
        bgr = resized.astype(np.float32)
        norm = (bgr - _MEAN_RGB) / _SCALE_RGB
    elif variant == "rgb_div255":
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = rgb
    elif variant == "swapped_mean_scale":
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - _SCALE_RGB) / _MEAN_RGB
    else:
        raise ValueError(variant)
    return np.expand_dims(np.transpose(norm, (2, 0, 1)), axis=0)


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_metrics(frame_bgr: np.ndarray, bbox: tuple[float, float, float, float]) -> dict:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    crop = crop_face_bgr(frame_bgr, bbox)
    if crop is None:
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return {
        "frame_w": w,
        "frame_h": h,
        "bbox_area_frac": (bw * bh) / (w * h),
        "crop_w": crop.shape[1],
        "crop_h": crop.shape[0],
        "crop_aspect": crop.shape[1] / crop.shape[0],
        "crop_mean_luma": float(np.mean(gray)),
        "crop_laplacian_var": laplacian_var(gray),
    }


def audit_integration(session) -> dict:
    det = LivenessDetector()
    rng = np.random.default_rng(42)
    fake = rng.integers(0, 256, (96, 88, 3), dtype=np.uint8)
    ours = det.preprocess_face_crop_bgr(fake)
    ref = omz_reference_preprocess_bgr(fake)
    max_diff = float(np.max(np.abs(ours - ref)))
    raw_checks = []
    for _ in range(5):
        fake2 = rng.integers(0, 256, (120, 100, 3), dtype=np.uint8)
        blob = det.preprocess_face_crop_bgr(fake2)
        inp = session.get_inputs()[0].name
        out = session.get_outputs()[0].name
        (raw,) = session.run([out], {inp: blob})
        vec = np.asarray(raw).reshape(-1)
        raw_checks.append(
            {
                "sum": float(np.sum(vec)),
                "min": float(np.min(vec)),
                "max": float(np.max(vec)),
            }
        )
    meta = {
        "input_name": session.get_inputs()[0].name,
        "input_shape": session.get_inputs()[0].shape,
        "output_name": session.get_outputs()[0].name,
        "output_shape": session.get_outputs()[0].shape,
        "providers": session.get_providers(),
    }
    return {
        "omz_reference_tensor_max_abs_diff_vs_detector": max_diff,
        "omz_reference_matches": max_diff == 0.0,
        "raw_output_samples": raw_checks,
        "onnx_metadata": meta,
        "class_mapping_documented": "index 0 REAL, index 1 SPOOF (OMZ README)",
    }


def audit_head_motion(run_dir: Path, session, liveness: LivenessDetector) -> dict:
    data = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    samples_b = data["phase_B_head_motion"]["samples"]
    b_max = max(s["index"] for s in samples_b) if samples_b else 0
    variants = [
        "omz_rgb",
        "bgr_converted_ir_style",
        "rgb_div255",
        "swapped_mean_scale",
        "interp_area",
    ]
    crop_mae: list[float] = []
    crop_shape_mismatch = 0
    bbox_redetect_drift: list[float] = []
    variant_scores: dict[str, list[float]] = {v: [] for v in variants}
    recorded_reals: list[float] = []
    yaw_abs: list[float] = []
    repro_live_vs_jpeg: list[float] = []
    frontal_reals: list[float] = []
    face_detector = load_face_detector_module().FaceDetector()

    for phase_key, phase_id in (("phase_A_stationary", "A"), ("phase_B_head_motion", "B")):
        for s in data[phase_key]["samples"]:
            if s.get("real_score") is None or not s.get("bbox"):
                continue
            frame_path = resolve_run_path(run_dir, s["frame_path"])
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            bbox = tuple(s["bbox"])
            recorded_reals.append(float(s["real_score"]))
            yaw = float(s["landmark_yaw_proxy"])
            yaw_abs.append(abs(yaw))
            bucket = classify_pose_bucket(phase_id, int(s["index"]), yaw, b_max)
            if bucket == "stationary_frontal":
                frontal_reals.append(float(s["real_score"]))

            crop_live = crop_face_bgr(frame, bbox)
            if crop_live is None:
                continue
            for v in variants:
                blob = variant_preprocess(crop_live, v)
                r, _ = run_onnx(session, blob)
                variant_scores[v].append(r)

            crop_rel = s.get("crop_path")
            if crop_rel:
                saved = cv2.imread(str(resolve_run_path(run_dir, crop_rel)))
                if saved is not None:
                    if saved.shape[:2] != crop_live.shape[:2]:
                        crop_shape_mismatch += 1
                    if saved.shape == crop_live.shape:
                        crop_mae.append(float(np.mean(np.abs(saved.astype(np.float32) - crop_live.astype(np.float32)))))
                    live_r = liveness.predict(crop_live).real_score
                    jpeg_r = liveness.predict(saved).real_score
                    repro_live_vs_jpeg.append(abs(live_r - jpeg_r))

            det = face_detector.detect(frame)
            if getattr(det.decision, "value", det.decision) == "FACE_OK" and det.faces:
                db = det.faces[0].bbox
                drift = max(abs(db[i] - bbox[i]) for i in range(4))
                bbox_redetect_drift.append(drift)

    def corr(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 3:
            return None
        return float(np.corrcoef(xs, ys)[0, 1])

    variant_means = {k: statistics.mean(v) if v else None for k, v in variant_scores.items()}
    return {
        "device": data.get("device_resolved"),
        "phase_A_mean_real_recorded_live": data["phase_A_stationary"]["stats"]["real_score_mean"],
        "phase_B_mean_real_recorded_live": data["phase_B_head_motion"]["stats"]["real_score_mean"],
        "stationary_frontal_mean_real": statistics.mean(frontal_reals) if frontal_reals else None,
        "model_argmax_spoof_phase_A": data["phase_A_stationary"]["stats"]["model_argmax_spoof"],
        "crop_recompute_vs_saved_jpeg_mean_abs_pixel": statistics.mean(crop_mae) if crop_mae else None,
        "crop_shape_mismatch_count": crop_shape_mismatch,
        "scrfd_redetect_max_corner_drift_px": max(bbox_redetect_drift) if bbox_redetect_drift else None,
        "scrfd_redetect_mean_corner_drift_px": statistics.mean(bbox_redetect_drift) if bbox_redetect_drift else None,
        "score_abs_diff_live_crop_vs_jpeg_saved_crop": {
            "mean": statistics.mean(repro_live_vs_jpeg) if repro_live_vs_jpeg else None,
            "max": max(repro_live_vs_jpeg) if repro_live_vs_jpeg else None,
            "n": len(repro_live_vs_jpeg),
        },
        "yaw_proxy_abs_corr_with_recorded_real": corr(yaw_abs, recorded_reals),
        "preprocess_variant_mean_real_on_same_crops": variant_means,
        "preprocess_variant_delta_vs_omz": {
            k: (variant_means[k] - variant_means["omz_rgb"])
            for k in variants
            if k != "omz_rgb" and variant_means.get(k) is not None
        },
    }


def audit_phone_video(video_path: Path, liveness: LivenessDetector, max_frames: int = 120) -> dict:
    if not video_path.is_file():
        return {"status": "missing", "path": str(video_path)}
    face_detector = load_face_detector_module().FaceDetector()
    cap = cv2.VideoCapture(str(video_path))
    reals: list[float] = []
    metrics: list[dict] = []
    step = 0
    while len(reals) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        step += 1
        if step % 6 != 0:
            continue
        det = face_detector.detect(frame)
        if getattr(det.decision, "value", det.decision) != "FACE_OK" or not det.faces:
            continue
        bbox = tuple(det.faces[0].bbox)
        crop = crop_face_bgr(frame, bbox)
        if crop is None:
            continue
        r = liveness.predict(crop).real_score
        reals.append(r)
        m = frame_metrics(frame, bbox)
        if m:
            metrics.append(m)
    cap.release()
    if not reals:
        return {"status": "no_face_ok", "path": str(video_path)}
    agg = {}
    for key in metrics[0]:
        agg[f"mean_{key}"] = statistics.mean([m[key] for m in metrics])
    return {
        "status": "ok",
        "path": str(video_path),
        "note": "Offline file decode (not Camo); historical mean ~0.98 REAL on full video",
        "sampled_face_ok_frames": len(reals),
        "mean_real": statistics.mean(reals),
        "median_real": statistics.median(reals),
        "p10_real": sorted(reals)[max(0, len(reals) // 10)],
        "metrics": agg,
    }


def audit_camo_phase_a_metrics(run_dir: Path) -> dict:
    data = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    rows = []
    for s in data["phase_A_stationary"]["samples"]:
        if s.get("real_score") is None or not s.get("bbox"):
            continue
        frame = cv2.imread(str(resolve_run_path(run_dir, s["frame_path"])))
        if frame is None:
            continue
        m = frame_metrics(frame, tuple(s["bbox"]))
        m["real_score"] = s["real_score"]
        rows.append(m)
    if not rows:
        return {}
    agg = {}
    for key in ("bbox_area_frac", "crop_mean_luma", "crop_laplacian_var", "crop_aspect"):
        agg[f"mean_{key}"] = statistics.mean([r[key] for r in rows])
    agg["mean_real_score"] = statistics.mean([r["real_score"] for r in rows])
    return agg


def main() -> None:
    out_dir = BASE_DIR / "diagnostic_output" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)

    liveness = LivenessDetector()
    session = ort.InferenceSession(
        str(liveness.model_path),
        providers=["CPUExecutionProvider"],
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "purpose": "liveness_accuracy_audit",
        "model": MODEL_NAME,
        "task1_integration": audit_integration(session),
        "task2_head_motion_crops": audit_head_motion(HEAD_RUN, session, liveness),
        "task3_camera_compare": {
            "camo_phase_A_metrics": audit_camo_phase_a_metrics(HEAD_RUN),
            "phone_video_sample": audit_phone_video(PHONE_VIDEO, liveness),
            "nazmy_single_frame_diagnostic": json.loads(NAZMY_DIAG.read_text(encoding="utf-8"))
            if NAZMY_DIAG.is_file()
            else {"status": "missing"},
        },
        "limitations": [
            "Head-motion JPEG replay lowers mn3 scores vs recorded live (see score_abs_diff).",
            "Phone video and Camo are different capture chains; not pixel-aligned.",
            "No SCREEN attack recording in repo.",
        ],
    }

    json_path = out_dir / "liveness_accuracy_audit.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

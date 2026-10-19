"""Offline research: anti-spoof-mn3 vs MiniFASNetV2 on shared diagnostic frames."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from crop import crop_face_bgr
from detector import LivenessDetector, MODEL_NAME as MN3_NAME
from diagnose_offline_crop_expansion import classify_pose_bucket, resolve_run_path
from face_bridge import load_face_detector_module
from research.minifasnetv2 import (
    CLASS_INDEX_REAL,
    CROP_SCALE,
    INPUT_SIZE,
    MODEL_NAME as MINIFAS_NAME,
    MiniFASNetV2Research,
)

BASE_DIR = Path(__file__).resolve().parent
REPRO_FRAME_BBOX_TOLERANCE = 0.25


@dataclass
class HeadMotionRow:
    phase: str
    index: int
    pose_bucket: str
    yaw_proxy: float
    recorded_mn3_real: float
    recorded_mn3_spoof: float
    recorded_mn3_class: str
    mn3: dict
    minifas: dict


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize_mn3(rows: list[HeadMotionRow]) -> dict:
    reals = [r["mn3"]["real_score"] for r in rows]
    spoofs = [r["mn3"]["spoof_score"] for r in rows]
    spoof_n = sum(1 for r in rows if r["mn3"]["predicted_class"] == "SPOOF")
    lat = [r["mn3"]["inference_ms"] for r in rows]
    return {
        "n": len(rows),
        "mean_real": statistics.mean(reals) if reals else None,
        "median_real": statistics.median(reals) if reals else None,
        "p10_real": _percentile(reals, 10) if reals else None,
        "mean_spoof": statistics.mean(spoofs) if spoofs else None,
        "argmax_spoof_count": spoof_n,
        "argmax_spoof_pct": 100.0 * spoof_n / len(rows) if rows else 0.0,
        "median_inference_ms": statistics.median(lat) if lat else None,
        "mean_inference_ms": statistics.mean(lat) if lat else None,
    }


def summarize_minifas(rows: list[HeadMotionRow]) -> dict:
    reals = [r["minifas"]["real_class_prob"] for r in rows]
    spoof_n = sum(1 for r in rows if r["minifas"]["predicted_class"] == "SPOOF")
    lat = [r["minifas"]["inference_ms"] for r in rows]
    return {
        "n": len(rows),
        "mean_real_class_prob": statistics.mean(reals) if reals else None,
        "median_real_class_prob": statistics.median(reals) if reals else None,
        "p10_real_class_prob": _percentile(reals, 10) if reals else None,
        "mean_spoof_mass": statistics.mean([r["minifas"]["spoof_mass"] for r in rows]) if rows else None,
        "argmax_spoof_count": spoof_n,
        "argmax_spoof_pct": 100.0 * spoof_n / len(rows) if rows else 0.0,
        "median_inference_ms": statistics.median(lat) if lat else None,
        "mean_inference_ms": statistics.mean(lat) if lat else None,
    }


def row_dict(phase, index, bucket, yaw, s, mn3_live, mf_live) -> HeadMotionRow:
    return HeadMotionRow(
        phase=phase,
        index=index,
        pose_bucket=bucket,
        yaw_proxy=yaw,
        recorded_mn3_real=float(s["real_score"]),
        recorded_mn3_spoof=float(s["spoof_score"]),
        recorded_mn3_class=s["predicted_class"],
        mn3={
            "real_score": mn3_live.real_score,
            "spoof_score": mn3_live.spoof_score,
            "predicted_class": mn3_live.predicted_class.value,
            "inference_ms": mn3_live.inference_ms,
            "preprocess": "OMZ anti-spoof-mn3: 1.0x SCRFD xyxy crop, resize 128x128, RGB mean/scale",
            "decision": "argmax REAL vs SPOOF (2-class)",
        },
        minifas={
            "softmax": list(mf_live.softmax),
            "real_class_prob": mf_live.real_class_prob,
            "spoof_mass": mf_live.spoof_mass,
            "predicted_class": mf_live.predicted_class.value,
            "argmax_label": mf_live.argmax_label,
            "inference_ms": mf_live.inference_ms,
            "preprocess": (
                f"Silent-Face CropImage scale={CROP_SCALE}, {INPUT_SIZE[1]}x{INPUT_SIZE[0]}, "
                "ToTensor CHW float 0-255"
            ),
            "decision": f"argmax; class {CLASS_INDEX_REAL} => REAL else SPOOF (official test.py)",
        },
    )


def evaluate_head_motion(run_dir: Path, mn3: LivenessDetector, minifas: MiniFASNetV2Research) -> dict:
    report_path = run_dir / "report.json"
    data = json.loads(report_path.read_text(encoding="utf-8"))
    samples_b = data["phase_B_head_motion"]["samples"]
    b_max_index = max(s["index"] for s in samples_b) if samples_b else 0

    scored: list[dict] = []
    repro: list[dict] = []
    missing_frames: list[str] = []

    for phase_key, phase_id in (("phase_A_stationary", "A"), ("phase_B_head_motion", "B")):
        for s in data[phase_key]["samples"]:
            if s.get("real_score") is None or not s.get("bbox"):
                continue
            frame_rel = s.get("frame_path")
            if not frame_rel:
                continue
            frame_path = resolve_run_path(run_dir, frame_rel)
            if not frame_path.is_file():
                missing_frames.append(frame_rel)
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                missing_frames.append(frame_rel)
                continue
            bbox = tuple(s["bbox"])
            yaw = float(s["landmark_yaw_proxy"])
            bucket = classify_pose_bucket(phase_id, int(s["index"]), yaw, b_max_index)

            crop_mn3 = crop_face_bgr(frame, bbox, expand_factor=1.0)
            if crop_mn3 is None:
                continue
            mn3_live = mn3.predict(crop_mn3)
            mf_live = minifas.predict_frame_bbox(frame, bbox)
            if mf_live is None:
                continue

            row = row_dict(phase_id, int(s["index"]), bucket, yaw, s, mn3_live, mf_live)
            scored.append(row.__dict__)
            repro.append(
                {
                    "phase": phase_id,
                    "index": s["index"],
                    "recorded_mn3_real": s["real_score"],
                    "recomputed_mn3_real": mn3_live.real_score,
                    "abs_diff": abs(mn3_live.real_score - float(s["real_score"])),
                }
            )

    buckets = sorted({r["pose_bucket"] for r in scored})
    by_bucket: dict[str, dict] = {}
    for bucket in buckets:
        rows = [r for r in scored if r["pose_bucket"] == bucket]
        by_bucket[bucket] = {
            "anti-spoof-mn3": summarize_mn3(rows),
            "MiniFASNetV2": summarize_minifas(rows),
        }

    return {
        "samples_scored": len(scored),
        "missing_frames": missing_frames,
        "repro_mn3_frame_bbox_jpeg": {
            "tolerance": REPRO_FRAME_BBOX_TOLERANCE,
            "n": len(repro),
            "within_tolerance": sum(1 for r in repro if r["abs_diff"] <= REPRO_FRAME_BBOX_TOLERANCE),
            "mean_abs_diff": statistics.mean([r["abs_diff"] for r in repro]) if repro else None,
            "note": "Recorded live mn3 scores preserved separately; JPEG frame replay vs live capture.",
        },
        "recorded_live_mn3_only": {
            "n": len(scored),
            "mean_real": statistics.mean([r["recorded_mn3_real"] for r in scored]),
            "argmax_spoof_count": sum(1 for r in scored if r["recorded_mn3_class"] == "SPOOF"),
        },
        "overall": {
            "anti-spoof-mn3": summarize_mn3(scored),
            "MiniFASNetV2": summarize_minifas(scored),
        },
        "by_pose_bucket": by_bucket,
        "per_sample": scored,
    }


def evaluate_static_image(
    label: str,
    image_path: Path,
    face_detector,
    mn3: LivenessDetector,
    minifas: MiniFASNetV2Research,
) -> dict | None:
    if not image_path.is_file():
        return {"label": label, "status": "missing", "path": str(image_path)}
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return {"label": label, "status": "unreadable", "path": str(image_path)}
    det = face_detector.detect(bgr)
    if getattr(det.decision, "value", det.decision) != "FACE_OK" or not det.faces:
        return {"label": label, "status": "no_face", "path": str(image_path)}
    bbox = tuple(det.faces[0].bbox)
    crop = crop_face_bgr(bgr, bbox, expand_factor=1.0)
    if crop is None:
        return {"label": label, "status": "crop_failed", "path": str(image_path)}
    mn3_r = mn3.predict(crop)
    mf_r = minifas.predict_frame_bbox(bgr, bbox)
    return {
        "label": label,
        "status": "ok",
        "path": str(image_path),
        "expected": label,
        "bbox_scrfd_xyxy": list(bbox),
        "anti-spoof-mn3": {
            "real_score": mn3_r.real_score,
            "spoof_score": mn3_r.spoof_score,
            "predicted_class": mn3_r.predicted_class.value,
            "inference_ms": mn3_r.inference_ms,
        },
        "MiniFASNetV2": {
            "softmax": list(mf_r.softmax) if mf_r else None,
            "real_class_prob": mf_r.real_class_prob if mf_r else None,
            "predicted_class": mf_r.predicted_class.value if mf_r else None,
            "inference_ms": mf_r.inference_ms if mf_r else None,
        },
    }


def evaluate_print_video(
    video_path: Path,
    face_detector,
    mn3: LivenessDetector,
    minifas: MiniFASNetV2Research,
    max_frames: int,
) -> dict:
    if not video_path.is_file():
        return {
            "status": "missing",
            "path": str(video_path),
            "note": "No local PRINT recording at expected path.",
        }
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "unreadable", "path": str(video_path)}

    mn3_spoof = 0
    mf_spoof = 0
    face_ok = 0
    mn3_reals: list[float] = []
    mf_reals: list[float] = []
    mn3_lat: list[float] = []
    mf_lat: list[float] = []

    while face_ok < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        det = face_detector.detect(frame)
        if getattr(det.decision, "value", det.decision) != "FACE_OK" or not det.faces:
            continue
        bbox = tuple(det.faces[0].bbox)
        crop = crop_face_bgr(frame, bbox, expand_factor=1.0)
        if crop is None:
            continue
        mn3_r = mn3.predict(crop)
        mf_r = minifas.predict_frame_bbox(frame, bbox)
        if mf_r is None:
            continue
        face_ok += 1
        mn3_reals.append(mn3_r.real_score)
        mf_reals.append(mf_r.real_class_prob)
        mn3_lat.append(mn3_r.inference_ms)
        mf_lat.append(mf_r.inference_ms)
        if mn3_r.predicted_class.value == "SPOOF":
            mn3_spoof += 1
        if mf_r.predicted_class.value == "SPOOF":
            mf_spoof += 1

    cap.release()
    return {
        "status": "ok",
        "path": str(video_path),
        "assumed_attack": "PRINT_or_physical_replay",
        "note": (
            "Historical video_results JSON shows ~100% mn3 SPOOF on this file; "
            "not a saved-frame paired test with head-motion diagnostic."
        ),
        "face_ok_frames_evaluated": face_ok,
        "max_frames_cap": max_frames,
        "anti-spoof-mn3": {
            "mean_real": statistics.mean(mn3_reals) if mn3_reals else None,
            "argmax_spoof_count": mn3_spoof,
            "argmax_spoof_pct": 100.0 * mn3_spoof / face_ok if face_ok else 0.0,
            "median_inference_ms": statistics.median(mn3_lat) if mn3_lat else None,
        },
        "MiniFASNetV2": {
            "mean_real_class_prob": statistics.mean(mf_reals) if mf_reals else None,
            "argmax_spoof_count": mf_spoof,
            "argmax_spoof_pct": 100.0 * mf_spoof / face_ok if face_ok else 0.0,
            "median_inference_ms": statistics.median(mf_lat) if mf_lat else None,
        },
    }


def build_table(payload: dict) -> str:
    lines = [
        "--- MiniFASNetV2 vs anti-spoof-mn3 (research only, production unchanged) ---",
        f"Head-motion run: {payload['run_dir']}",
        f"Paired samples (same JPEG frame + SCRFD bbox): {payload['head_motion']['samples_scored']}",
        "",
        "Model preprocessing (native, not shared):",
        f"  mn3: 1.0x SCRFD crop, 128x128, OMZ RGB mean/scale, 2-class",
        f"  MiniFAS: Silent-Face scale {CROP_SCALE}, {INPUT_SIZE[1]}x{INPUT_SIZE[0]}, 3-class softmax",
        "",
        "Overall (genuine live person, JPEG replay):",
        "| Model | mean REAL signal | argmax SPOOF | median ms |",
        "|---|---:|---:|---:|",
    ]
    o_mn3 = payload["head_motion"]["overall"]["anti-spoof-mn3"]
    o_mf = payload["head_motion"]["overall"]["MiniFASNetV2"]
    lines.append(
        f"| anti-spoof-mn3 | mean_real={o_mn3['mean_real']:.4f} | "
        f"{o_mn3['argmax_spoof_count']}/{o_mn3['n']} ({o_mn3['argmax_spoof_pct']:.1f}%) | "
        f"{o_mn3['median_inference_ms']:.2f} |"
    )
    lines.append(
        f"| MiniFASNetV2 | mean class1={o_mf['mean_real_class_prob']:.4f} | "
        f"{o_mf['argmax_spoof_count']}/{o_mf['n']} ({o_mf['argmax_spoof_pct']:.1f}%) | "
        f"{o_mf['median_inference_ms']:.2f} |"
    )
    lines.append("")
    lines.append("By pose bucket (mean REAL / class1 prob, SPOOF argmax %):")
    lines.append("| Pose | mn3 mean REAL | mn3 SPOOF% | MiniFAS class1 | MiniFAS SPOOF% |")
    lines.append("|---|---:|---:|---:|---:|")
    for bucket, stats in payload["head_motion"]["by_pose_bucket"].items():
        m = stats["anti-spoof-mn3"]
        f = stats["MiniFASNetV2"]
        lines.append(
            f"| {bucket} | {m['mean_real']:.3f} | {m['argmax_spoof_pct']:.0f}% | "
            f"{f['mean_real_class_prob']:.3f} | {f['argmax_spoof_pct']:.0f}% |"
        )
    lines.append("")
    lines.append(
        f"Recorded live mn3 (not JPEG replay): mean_real="
        f"{payload['head_motion']['recorded_live_mn3_only']['mean_real']:.4f}, "
        f"live SPOOF argmax={payload['head_motion']['recorded_live_mn3_only']['argmax_spoof_count']}"
    )
    lines.append(
        f"JPEG replay mn3 repro within {REPRO_FRAME_BBOX_TOLERANCE}: "
        f"{payload['head_motion']['repro_mn3_frame_bbox_jpeg']['within_tolerance']}/"
        f"{payload['head_motion']['repro_mn3_frame_bbox_jpeg']['n']}"
    )
    lines.append("")
    lines.append("--- Attack / static (not paired with head-motion frames) ---")
    for item in payload.get("static_eval", []):
        if item.get("status") != "ok":
            lines.append(f"{item.get('label')}: {item.get('status')} ({item.get('path', '')})")
            continue
        lines.append(
            f"{item['label']}: mn3 {item['anti-spoof-mn3']['predicted_class']} "
            f"(R={item['anti-spoof-mn3']['real_score']:.3f}) | "
            f"MiniFAS {item['MiniFASNetV2']['predicted_class']} "
            f"(c1={item['MiniFASNetV2']['real_class_prob']:.3f})"
        )
    pv = payload.get("print_video_eval", {})
    if pv.get("status") == "ok":
        lines.append(
            f"PRINT-like video ({pv['face_ok_frames_evaluated']} frames): "
            f"mn3 SPOOF {pv['anti-spoof-mn3']['argmax_spoof_pct']:.1f}% | "
            f"MiniFAS SPOOF {pv['MiniFASNetV2']['argmax_spoof_pct']:.1f}%"
        )
    else:
        lines.append(f"PRINT video: {pv.get('status', 'skipped')}")
    lines.append(f"SCREEN attack: {payload.get('screen_attack', {}).get('status', 'not evaluated')}")
    lines.append("")
    lines.append(payload["security_note"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research compare mn3 vs MiniFASNetV2")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=BASE_DIR / "diagnostic_head_motion" / "20261015T102037Z",
    )
    parser.add_argument(
        "--print-video",
        type=Path,
        default=Path(r"C:\Users\smnam\Downloads\test video.mp4"),
    )
    parser.add_argument("--print-max-frames", type=int, default=400)
    parser.add_argument("--skip-print-video", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "report.json").is_file():
        print(f"[ERROR] Missing report in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Loading models (research)...", flush=True)
    t0 = time.perf_counter()
    mn3 = LivenessDetector()
    minifas = MiniFASNetV2Research()
    face_detector = load_face_detector_module().FaceDetector()
    print(f"[INFO] Ready in {time.perf_counter() - t0:.1f}s", flush=True)

    head_motion = evaluate_head_motion(run_dir, mn3, minifas)

    hmac_dir = BASE_DIR.parent / "hmac" / "test_images"
    static_eval = [
        evaluate_static_image("genuine_static", hmac_dir / "original_1.jpg", face_detector, mn3, minifas),
        evaluate_static_image("print_like_static", hmac_dir / "fake_image.jpg", face_detector, mn3, minifas),
    ]

    print_video_eval = (
        {"status": "skipped", "reason": "--skip-print-video"}
        if args.skip_print_video
        else evaluate_print_video(args.print_video, face_detector, mn3, minifas, args.print_max_frames)
    )

    payload = {
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "research_only": True,
        "production_model_unchanged": MN3_NAME,
        "alternative_model": MINIFAS_NAME,
        "run_dir": str(run_dir),
        "checkpoint": str(minifas.checkpoint_path),
        "head_motion": head_motion,
        "static_eval": static_eval,
        "print_video_eval": print_video_eval,
        "screen_attack": {
            "status": "no_saved_screen_recording",
            "note": (
                "No SCREEN attack video or saved diagnostic frames in repo. "
                "Use live_liveness.py --scenario SCREEN locally to collect."
            ),
        },
        "security_note": (
            "Scores are native and uncalibrated across models; higher genuine REAL "
            "does not imply better PRINT/SCREEN resistance. Compare attack metrics explicitly."
        ),
    }

    out_dir = run_dir / "minifas_vs_mn3"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = build_table(payload)
    table_path = out_dir / "comparison_table.txt"
    table_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"\nJSON: {json_path}")
    sys.exit(0 if not head_motion["missing_frames"] else 1)


if __name__ == "__main__":
    main()

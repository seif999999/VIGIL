"""Research benchmark: CVPR2024 MobileNetV3-Small vs anti-spoof-mn3 (production unchanged)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from crop import crop_face_bgr
from detector import LivenessDetector, MODEL_NAME as MN3_NAME
from diagnose_offline_crop_expansion import classify_pose_bucket, resolve_run_path
from face_bridge import load_face_detector_module
from research.cvpr2024_fas import (
    INPUT_SIZE,
    MODEL_NAME as CVPR_NAME,
    Cvpr2024MobileNetV3Research,
)

BASE_DIR = Path(__file__).resolve().parent
HEAD_RUN = BASE_DIR / "diagnostic_head_motion" / "20261015T102037Z"
PHONE_VIDEO = BASE_DIR.parent / "face_detection" / "test video.mp4"
PRINT_VIDEO = Path(r"C:\Users\smnam\Downloads\test video.mp4")


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _summ_mn3(rows: list[dict]) -> dict:
    reals = [r["mn3"]["real_score"] for r in rows]
    spoof_n = sum(1 for r in rows if r["mn3"]["predicted_class"] == "SPOOF")
    lat = [r["mn3"]["inference_ms"] for r in rows]
    return {
        "n": len(rows),
        "mean_real": statistics.mean(reals) if reals else None,
        "median_real": statistics.median(reals) if reals else None,
        "p10_real": _pct(reals, 10),
        "p90_real": _pct(reals, 90),
        "argmax_spoof_count": spoof_n,
        "argmax_spoof_pct": 100.0 * spoof_n / len(rows) if rows else 0.0,
        "argmax_real_pct": 100.0 * (len(rows) - spoof_n) / len(rows) if rows else 0.0,
        "median_inference_ms": statistics.median(lat) if lat else None,
        "mean_inference_ms": statistics.mean(lat) if lat else None,
    }


def _summ_cvpr(rows: list[dict], key: str) -> dict:
    lives = [r[key]["live_prob"] for r in rows]
    spoof_n = sum(1 for r in rows if r[key]["predicted_class"] == "SPOOF")
    lat = [r[key]["inference_ms"] for r in rows]
    return {
        "n": len(rows),
        "mean_live_prob": statistics.mean(lives) if lives else None,
        "median_live_prob": statistics.median(lives) if lives else None,
        "p10_live_prob": _pct(lives, 10),
        "p90_live_prob": _pct(lives, 90),
        "argmax_spoof_count": spoof_n,
        "argmax_spoof_pct": 100.0 * spoof_n / len(rows) if rows else 0.0,
        "argmax_real_pct": 100.0 * (len(rows) - spoof_n) / len(rows) if rows else 0.0,
        "median_inference_ms": statistics.median(lat) if lat else None,
        "mean_inference_ms": statistics.mean(lat) if lat else None,
        "input_mode": rows[0][key]["input_mode"] if rows else None,
        "decision_rule": "argmax; class0=LIVE class1=SPOOF (official)",
    }


def eval_head_motion(run_dir: Path, mn3: LivenessDetector, cvpr: Cvpr2024MobileNetV3Research) -> dict:
    data = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    samples_b = data["phase_B_head_motion"]["samples"]
    b_max = max(s["index"] for s in samples_b) if samples_b else 0
    scored: list[dict] = []
    missing: list[str] = []

    for phase_key, phase_id in (("phase_A_stationary", "A"), ("phase_B_head_motion", "B")):
        for s in data[phase_key]["samples"]:
            if s.get("real_score") is None or not s.get("bbox"):
                continue
            frame_rel = s.get("frame_path")
            if not frame_rel:
                continue
            frame_path = resolve_run_path(run_dir, frame_rel)
            if not frame_path.is_file():
                missing.append(frame_rel)
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                missing.append(frame_rel)
                continue
            bbox = tuple(s["bbox"])
            crop = crop_face_bgr(frame, bbox, expand_factor=1.0)
            if crop is None:
                continue
            mn3_r = mn3.predict(crop)
            cvpr_full = cvpr.predict_bgr(frame, input_mode="full_frame_official")
            cvpr_crop = cvpr.predict_bgr(crop, input_mode="scrfd_face_crop")
            yaw = float(s["landmark_yaw_proxy"])
            scored.append(
                {
                    "phase": phase_id,
                    "index": int(s["index"]),
                    "pose_bucket": classify_pose_bucket(phase_id, int(s["index"]), yaw, b_max),
                    "yaw_proxy": yaw,
                    "recorded_live_mn3_real": float(s["real_score"]),
                    "mn3": {
                        "real_score": mn3_r.real_score,
                        "spoof_score": mn3_r.spoof_score,
                        "predicted_class": mn3_r.predicted_class.value,
                        "inference_ms": mn3_r.inference_ms,
                        "preprocess": "OMZ RGB mean/scale, 128x128, SCRFD 1.0x crop",
                    },
                    "cvpr_full": {
                        "live_prob": cvpr_full.live_prob,
                        "spoof_prob": cvpr_full.spoof_prob,
                        "predicted_class": cvpr_full.predicted_class.value,
                        "inference_ms": cvpr_full.inference_ms,
                        "input_mode": cvpr_full.input_mode,
                        "preprocess": f"Resize {INPUT_SIZE}, BGR2RGB, ImageNet norm, full frame",
                    },
                    "cvpr_crop": {
                        "live_prob": cvpr_crop.live_prob,
                        "spoof_prob": cvpr_crop.spoof_prob,
                        "predicted_class": cvpr_crop.predicted_class.value,
                        "inference_ms": cvpr_crop.inference_ms,
                        "input_mode": cvpr_crop.input_mode,
                        "preprocess": f"Resize {INPUT_SIZE}, BGR2RGB, ImageNet norm, SCRFD crop",
                    },
                }
            )

    buckets = sorted({r["pose_bucket"] for r in scored})
    by_bucket = {}
    for b in buckets:
        rows = [r for r in scored if r["pose_bucket"] == b]
        by_bucket[b] = {
            "anti-spoof-mn3": _summ_mn3(rows),
            "cvpr_full_frame": _summ_cvpr(rows, "cvpr_full"),
            "cvpr_scrfd_crop": _summ_cvpr(rows, "cvpr_crop"),
        }
    return {
        "run_dir": str(run_dir),
        "device": data.get("device_resolved"),
        "samples_scored": len(scored),
        "missing_frames": missing,
        "note": (
            "Camo JPEG frames + SCRFD bboxes from diagnostic. "
            "Recorded live mn3 scores listed separately where present; "
            "JPEG replay may shift mn3 slightly."
        ),
        "overall": {
            "anti-spoof-mn3": _summ_mn3(scored),
            "cvpr_full_frame": _summ_cvpr(scored, "cvpr_full"),
            "cvpr_scrfd_crop": _summ_cvpr(scored, "cvpr_crop"),
        },
        "by_pose_bucket": by_bucket,
        "per_sample": scored,
    }


def eval_video(
    video_path: Path,
    label: str,
    mn3: LivenessDetector,
    cvpr: Cvpr2024MobileNetV3Research,
    face_detector,
    face_mod,
    *,
    max_face_ok: int,
    frame_stride: int,
) -> dict:
    if not video_path.is_file():
        return {"status": "missing", "path": str(video_path), "label": label}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "unreadable", "path": str(video_path), "label": label}
    rows: list[dict] = []
    idx = 0
    while len(rows) < max_face_ok:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        idx += 1
        if idx % frame_stride != 0:
            continue
        det = face_detector.detect(frame)
        if det.decision != face_mod.FaceCountDecision.FACE_OK or not det.faces:
            continue
        bbox = tuple(det.faces[0].bbox)
        crop = crop_face_bgr(frame, bbox)
        if crop is None:
            continue
        mn3_r = mn3.predict(crop)
        cvpr_full = cvpr.predict_bgr(frame, input_mode="full_frame_official")
        cvpr_crop = cvpr.predict_bgr(crop, input_mode="scrfd_face_crop")
        rows.append(
            {
                "frame_index": idx,
                "mn3": {
                    "real_score": mn3_r.real_score,
                    "spoof_score": mn3_r.spoof_score,
                    "predicted_class": mn3_r.predicted_class.value,
                    "inference_ms": mn3_r.inference_ms,
                },
                "cvpr_full": {
                    "live_prob": cvpr_full.live_prob,
                    "spoof_prob": cvpr_full.spoof_prob,
                    "predicted_class": cvpr_full.predicted_class.value,
                    "inference_ms": cvpr_full.inference_ms,
                    "input_mode": cvpr_full.input_mode,
                },
                "cvpr_crop": {
                    "live_prob": cvpr_crop.live_prob,
                    "spoof_prob": cvpr_crop.spoof_prob,
                    "predicted_class": cvpr_crop.predicted_class.value,
                    "inference_ms": cvpr_crop.inference_ms,
                    "input_mode": cvpr_crop.input_mode,
                },
            }
        )
    cap.release()
    return {
        "status": "ok",
        "label": label,
        "path": str(video_path),
        "face_ok_frames": len(rows),
        "anti-spoof-mn3": _summ_mn3(rows),
        "cvpr_full_frame": _summ_cvpr(rows, "cvpr_full"),
        "cvpr_scrfd_crop": _summ_cvpr(rows, "cvpr_crop"),
    }


def build_table(payload: dict) -> str:
    lines = [
        "--- CVPR2024 MobileNetV3-Small vs anti-spoof-mn3 (research only) ---",
        "Production baseline unchanged. Scores are native / uncalibrated across models.",
        f"CVPR checkpoint: {payload['cvpr_checkpoint']}",
        f"CVPR weight source: {payload['cvpr_meta'].get('weight_source')} epoch={payload['cvpr_meta'].get('epoch')}",
        "",
        "Official CVPR mapping: live=0, spoof=1; recommended input=full image for this weight.",
        "Also reported: SCRFD face-crop path (VIGIL-compatible, not official full-frame).",
        "",
    ]
    hm = payload["head_motion_camo"]
    o = hm["overall"]
    lines += [
        f"Camo head-motion diagnostic ({hm['device']}, n={hm['samples_scored']} JPEG frames):",
        "| Mode | mean live/REAL | SPOOF argmax % | median ms |",
        "|---|---:|---:|---:|",
        f"| mn3 SCRFD crop | {o['anti-spoof-mn3']['mean_real']:.4f} | "
        f"{o['anti-spoof-mn3']['argmax_spoof_pct']:.1f}% | {o['anti-spoof-mn3']['median_inference_ms']:.2f} |",
        f"| CVPR full frame | {o['cvpr_full_frame']['mean_live_prob']:.4f} | "
        f"{o['cvpr_full_frame']['argmax_spoof_pct']:.1f}% | {o['cvpr_full_frame']['median_inference_ms']:.2f} |",
        f"| CVPR SCRFD crop | {o['cvpr_scrfd_crop']['mean_live_prob']:.4f} | "
        f"{o['cvpr_scrfd_crop']['argmax_spoof_pct']:.1f}% | {o['cvpr_scrfd_crop']['median_inference_ms']:.2f} |",
        "",
        "By pose (mean live/REAL | SPOOF%):",
        "| Pose | mn3 | CVPR full | CVPR crop |",
        "|---|---:|---:|---:|",
    ]
    for bucket, stats in hm["by_pose_bucket"].items():
        lines.append(
            f"| {bucket} | {stats['anti-spoof-mn3']['mean_real']:.3f}/"
            f"{stats['anti-spoof-mn3']['argmax_spoof_pct']:.0f}% | "
            f"{stats['cvpr_full_frame']['mean_live_prob']:.3f}/"
            f"{stats['cvpr_full_frame']['argmax_spoof_pct']:.0f}% | "
            f"{stats['cvpr_scrfd_crop']['mean_live_prob']:.3f}/"
            f"{stats['cvpr_scrfd_crop']['argmax_spoof_pct']:.0f}% |"
        )

    for key, title in (
        ("phone_genuine_video", "Phone genuine video"),
        ("print_attack_video", "PRINT-like attack video"),
    ):
        v = payload.get(key) or {}
        lines.append("")
        if v.get("status") != "ok":
            lines.append(f"{title}: {v.get('status')} ({v.get('path')})")
            continue
        lines.append(f"{title} ({v['face_ok_frames']} FACE_OK frames):")
        lines.append(
            f"  mn3: mean_REAL={v['anti-spoof-mn3']['mean_real']:.4f} "
            f"SPOOF%={v['anti-spoof-mn3']['argmax_spoof_pct']:.1f} "
            f"median_ms={v['anti-spoof-mn3']['median_inference_ms']:.2f}"
        )
        lines.append(
            f"  CVPR full: mean_live={v['cvpr_full_frame']['mean_live_prob']:.4f} "
            f"SPOOF%={v['cvpr_full_frame']['argmax_spoof_pct']:.1f} "
            f"median_ms={v['cvpr_full_frame']['median_inference_ms']:.2f}"
        )
        lines.append(
            f"  CVPR crop: mean_live={v['cvpr_scrfd_crop']['mean_live_prob']:.4f} "
            f"SPOOF%={v['cvpr_scrfd_crop']['argmax_spoof_pct']:.1f} "
            f"median_ms={v['cvpr_scrfd_crop']['median_inference_ms']:.2f}"
        )

    lines += [
        "",
        f"SCREEN attack: {payload['screen_attack']['status']}",
        payload["screen_attack"]["note"],
        "",
        payload["security_note"],
        "",
        "Answers:",
    ]
    for q, a in payload["answers"].items():
        lines.append(f"- {q}: {a}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark CVPR2024 FAS vs mn3")
    parser.add_argument("--run-dir", type=Path, default=HEAD_RUN)
    parser.add_argument("--phone-video", type=Path, default=PHONE_VIDEO)
    parser.add_argument("--print-video", type=Path, default=PRINT_VIDEO)
    parser.add_argument("--max-face-ok", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    print("[INFO] Loading models...", flush=True)
    t0 = time.perf_counter()
    mn3 = LivenessDetector()
    cvpr = Cvpr2024MobileNetV3Research()
    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    print(f"[INFO] Ready in {time.perf_counter() - t0:.1f}s", flush=True)

    print("[INFO] Camo head-motion paired frames...", flush=True)
    head = eval_head_motion(args.run_dir.resolve(), mn3, cvpr)

    if args.skip_videos:
        phone = {"status": "skipped"}
        print_v = {"status": "skipped"}
    else:
        print("[INFO] Phone genuine video...", flush=True)
        phone = eval_video(
            args.phone_video.resolve(),
            "genuine_phone_file",
            mn3,
            cvpr,
            face_detector,
            face_mod,
            max_face_ok=args.max_face_ok,
            frame_stride=args.frame_stride,
        )
        print("[INFO] PRINT-like video...", flush=True)
        print_v = eval_video(
            args.print_video.resolve(),
            "print_like_attack",
            mn3,
            cvpr,
            face_detector,
            face_mod,
            max_face_ok=args.max_face_ok,
            frame_stride=args.frame_stride,
        )

    # Derive concise answers from available numbers
    hm_o = head["overall"]
    camo_mn3_spoof = hm_o["anti-spoof-mn3"]["argmax_spoof_pct"]
    camo_cvpr_full_spoof = hm_o["cvpr_full_frame"]["argmax_spoof_pct"]
    camo_cvpr_crop_spoof = hm_o["cvpr_scrfd_crop"]["argmax_spoof_pct"]
    stationary = head["by_pose_bucket"].get("stationary_frontal", {})
    strong = head["by_pose_bucket"].get("strong_side_profile", {})

    print_spoof_mn3 = (print_v.get("anti-spoof-mn3") or {}).get("argmax_spoof_pct")
    print_spoof_cvpr = (print_v.get("cvpr_full_frame") or {}).get("argmax_spoof_pct")

    answers = {
        "genuine_more_reliable": (
            f"On Camo JPEG set, CVPR full-frame SPOOF argmax {camo_cvpr_full_spoof:.1f}% "
            f"vs mn3 {camo_mn3_spoof:.1f}%; CVPR SCRFD-crop SPOOF {camo_cvpr_crop_spoof:.1f}%. "
            "Compare live/REAL means only within each model (uncalibrated)."
        ),
        "print_rejection": (
            f"PRINT video SPOOF%: mn3={print_spoof_mn3}, CVPR_full={print_spoof_cvpr}, "
            f"CVPR_crop={(print_v.get('cvpr_scrfd_crop') or {}).get('argmax_spoof_pct')} "
            "(higher SPOOF% is better for attacks)."
            if print_v.get("status") == "ok"
            else f"PRINT evaluation: {print_v.get('status')}"
        ),
        "head_motion": (
            f"strong_side mn3 REAL mean="
            f"{(strong.get('anti-spoof-mn3') or {}).get('mean_real')}; "
            f"CVPR_full live mean="
            f"{(strong.get('cvpr_full_frame') or {}).get('mean_live_prob')}; "
            f"stationary mn3="
            f"{(stationary.get('anti-spoof-mn3') or {}).get('mean_real')}, "
            f"CVPR_full="
            f"{(stationary.get('cvpr_full_frame') or {}).get('mean_live_prob')}."
        ),
        "latency": (
            f"median ms Camo: mn3={hm_o['anti-spoof-mn3']['median_inference_ms']:.2f}, "
            f"CVPR_full={hm_o['cvpr_full_frame']['median_inference_ms']:.2f}, "
            f"CVPR_crop={hm_o['cvpr_scrfd_crop']['median_inference_ms']:.2f}."
        ),
        "worth_further": (
            "Judge from PRINT SPOOF% vs genuine Camo SPOOF% trade-off in this report; "
            "do not replace production without SCREEN evidence and threshold redesign."
        ),
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "research_only": True,
        "production_model_unchanged": MN3_NAME,
        "candidate_model": CVPR_NAME,
        "cvpr_checkpoint": str(cvpr.checkpoint_path),
        "cvpr_meta": cvpr.meta,
        "head_motion_camo": head,
        "phone_genuine_video": phone,
        "print_attack_video": print_v,
        "screen_attack": {
            "status": "no_saved_screen_recording",
            "note": "No SCREEN replay video in repo; do not invent SCREEN results.",
        },
        "security_note": (
            "Native scores are not calibrated across models. "
            "Higher genuine live probability alone does not justify replacing mn3; "
            "PRINT/SCREEN attack rejection must remain adequate."
        ),
        "answers": answers,
    }

    out_dir = args.run_dir.resolve() / "cvpr2024_vs_mn3"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop per-sample from a slim table copy? Keep full JSON.
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = build_table(payload)
    table_path = out_dir / "comparison_table.txt"
    table_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"\nJSON: {json_path}")
    sys.exit(0 if not head["missing_frames"] else 1)


if __name__ == "__main__":
    main()

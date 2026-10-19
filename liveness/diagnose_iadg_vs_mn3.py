"""Research benchmark: IADG (CVPR 2023) vs anti-spoof-mn3. Production unchanged."""

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
from decision_policy import ProvisionalDecisionPolicy, classify_provisional_scores
from detector import LivenessDetector, MODEL_NAME as MN3_NAME
from diagnose_offline_crop_expansion import classify_pose_bucket, resolve_run_path
from face_bridge import load_face_detector_module
from research.iadg_fas import MODEL_NAME as IADG_NAME, IadgResearch

BASE_DIR = Path(__file__).resolve().parent
HEAD_RUNS = [
    BASE_DIR / "diagnostic_head_motion" / "20261015T102037Z",
    BASE_DIR / "diagnostic_head_motion" / "20261015T131532Z",
]
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
    prov: dict[str, int] = {}
    for r in rows:
        k = r["mn3"].get("provisional") or "n/a"
        prov[k] = prov.get(k, 0) + 1
    return {
        "n": len(rows),
        "mean_real": statistics.mean(reals) if reals else None,
        "median_real": statistics.median(reals) if reals else None,
        "p10_real": _pct(reals, 10),
        "p90_real": _pct(reals, 90),
        "argmax_spoof_count": spoof_n,
        "argmax_spoof_pct": 100.0 * spoof_n / len(rows) if rows else 0.0,
        "argmax_real_pct": 100.0 * (len(rows) - spoof_n) / len(rows) if rows else 0.0,
        "provisional_counts": prov,
        "median_inference_ms": statistics.median(lat) if lat else None,
        "mean_inference_ms": statistics.mean(lat) if lat else None,
    }


def _summ_iadg(rows: list[dict]) -> dict:
    lives = [r["iadg"]["live_score"] for r in rows]
    spoof_n = sum(1 for r in rows if r["iadg"]["predicted_class"] == "SPOOF")
    lat = [r["iadg"]["inference_ms"] for r in rows]
    return {
        "n": len(rows),
        "mean_live_score": statistics.mean(lives) if lives else None,
        "median_live_score": statistics.median(lives) if lives else None,
        "p10_live_score": _pct(lives, 10),
        "p90_live_score": _pct(lives, 90),
        "argmax_spoof_count": spoof_n,
        "argmax_spoof_pct": 100.0 * spoof_n / len(rows) if rows else 0.0,
        "argmax_real_pct": 100.0 * (len(rows) - spoof_n) / len(rows) if rows else 0.0,
        "median_inference_ms": statistics.median(lat) if lat else None,
        "mean_inference_ms": statistics.mean(lat) if lat else None,
        "decision_rule": "cls argmax (0=LIVE,1=SPOOF); live_score=cls0+depth_mean (uncalibrated)",
    }


def eval_camo_run(
    run_dir: Path,
    mn3: LivenessDetector,
    iadg: IadgResearch,
    face_detector,
    face_mod,
    policy: ProvisionalDecisionPolicy,
) -> dict:
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        return {"status": "missing", "path": str(run_dir)}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    samples_b = data["phase_B_head_motion"]["samples"]
    b_max = max(s["index"] for s in samples_b) if samples_b else 0
    scored: list[dict] = []
    missing: list[str] = []
    landmark_crops = 0
    bbox_crops = 0

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
            crop_mn3 = crop_face_bgr(frame, bbox, expand_factor=1.0)
            if crop_mn3 is None:
                continue
            mn3_r = mn3.predict(crop_mn3)
            # Re-detect for official 5-landmark crop on same JPEG frame.
            det = face_detector.detect(frame)
            landmarks = None
            if (
                det.decision == face_mod.FaceCountDecision.FACE_OK
                and det.faces
            ):
                landmarks = det.faces[0].landmarks
            iadg_r = iadg.predict_frame(frame, bbox, landmarks, prefer_landmarks=True)
            if iadg_r is None:
                continue
            if iadg_r.crop_mode == "landmarks_margin":
                landmark_crops += 1
            else:
                bbox_crops += 1
            yaw = float(s["landmark_yaw_proxy"])
            scored.append(
                {
                    "phase": phase_id,
                    "index": int(s["index"]),
                    "pose_bucket": classify_pose_bucket(
                        phase_id, int(s["index"]), yaw, b_max
                    ),
                    "yaw_proxy": yaw,
                    "recorded_live_mn3_real": float(s["real_score"]),
                    "mn3": {
                        "real_score": mn3_r.real_score,
                        "spoof_score": mn3_r.spoof_score,
                        "predicted_class": mn3_r.predicted_class.value,
                        "provisional": classify_provisional_scores(
                            mn3_r.real_score, mn3_r.spoof_score, policy
                        ).value,
                        "inference_ms": mn3_r.inference_ms,
                    },
                    "iadg": {
                        "live_score": iadg_r.live_score,
                        "cls_live_prob": iadg_r.cls_live_prob,
                        "depth_mean": iadg_r.depth_mean,
                        "spoof_cls_prob": iadg_r.spoof_cls_prob,
                        "predicted_class": iadg_r.predicted_class.value,
                        "argmax_cls": iadg_r.argmax_cls,
                        "inference_ms": iadg_r.inference_ms,
                        "crop_mode": iadg_r.crop_mode,
                    },
                }
            )

    buckets = sorted({r["pose_bucket"] for r in scored})
    by_bucket = {
        b: {
            "anti-spoof-mn3": _summ_mn3([r for r in scored if r["pose_bucket"] == b]),
            "IADG": _summ_iadg([r for r in scored if r["pose_bucket"] == b]),
        }
        for b in buckets
    }
    return {
        "status": "ok",
        "path": str(run_dir),
        "device": data.get("device_resolved"),
        "samples_scored": len(scored),
        "missing_frames": missing,
        "iadg_crop_modes": {
            "landmarks_margin": landmark_crops,
            "bbox_margin": bbox_crops,
        },
        "overall": {
            "anti-spoof-mn3": _summ_mn3(scored),
            "IADG": _summ_iadg(scored),
        },
        "by_pose_bucket": by_bucket,
        "per_sample": scored,
    }


def eval_video(
    video_path: Path,
    label: str,
    mn3: LivenessDetector,
    iadg: IadgResearch,
    face_detector,
    face_mod,
    policy: ProvisionalDecisionPolicy,
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
        face = det.faces[0]
        bbox = tuple(face.bbox)
        crop = crop_face_bgr(frame, bbox)
        if crop is None:
            continue
        mn3_r = mn3.predict(crop)
        iadg_r = iadg.predict_frame(frame, bbox, face.landmarks, prefer_landmarks=True)
        if iadg_r is None:
            continue
        rows.append(
            {
                "frame_index": idx,
                "mn3": {
                    "real_score": mn3_r.real_score,
                    "spoof_score": mn3_r.spoof_score,
                    "predicted_class": mn3_r.predicted_class.value,
                    "provisional": classify_provisional_scores(
                        mn3_r.real_score, mn3_r.spoof_score, policy
                    ).value,
                    "inference_ms": mn3_r.inference_ms,
                },
                "iadg": {
                    "live_score": iadg_r.live_score,
                    "cls_live_prob": iadg_r.cls_live_prob,
                    "depth_mean": iadg_r.depth_mean,
                    "predicted_class": iadg_r.predicted_class.value,
                    "inference_ms": iadg_r.inference_ms,
                    "crop_mode": iadg_r.crop_mode,
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
        "IADG": _summ_iadg(rows),
    }


def build_table(payload: dict) -> str:
    lines = [
        "--- IADG (CVPR 2023 OCI2M) vs anti-spoof-mn3 (research only) ---",
        "Production baseline unchanged. Scores are native / uncalibrated across models.",
        f"IADG checkpoint: {payload['iadg_checkpoint']}",
        f"IADG protocol: {payload['iadg_meta'].get('protocol')} epoch={payload['iadg_meta'].get('epoch')}",
        "IADG score: softmax(cls)[:,0] + mean(depth); argmax class0=LIVE class1=SPOOF",
        "IADG crop: 5-landmark square + margin 0.7 (official); bbox+margin fallback",
        "",
    ]
    for run in payload.get("camo_runs", []):
        if run.get("status") != "ok":
            lines.append(f"Camo run missing: {run.get('path')}")
            continue
        o = run["overall"]
        lines += [
            f"Camo run {Path(run['path']).name} ({run.get('device')}, n={run['samples_scored']}):",
            f"  mn3: mean_REAL={o['anti-spoof-mn3']['mean_real']:.4f} "
            f"SPOOF%={o['anti-spoof-mn3']['argmax_spoof_pct']:.1f} "
            f"provisional={o['anti-spoof-mn3']['provisional_counts']} "
            f"median_ms={o['anti-spoof-mn3']['median_inference_ms']:.1f}",
            f"  IADG: mean_live={o['IADG']['mean_live_score']:.4f} "
            f"SPOOF%={o['IADG']['argmax_spoof_pct']:.1f} "
            f"median_ms={o['IADG']['median_inference_ms']:.1f} "
            f"crops={run['iadg_crop_modes']}",
            "  By pose (mean | SPOOF%):",
        ]
        for bucket, stats in run["by_pose_bucket"].items():
            lines.append(
                f"    {bucket}: mn3 {stats['anti-spoof-mn3']['mean_real']:.3f}/"
                f"{stats['anti-spoof-mn3']['argmax_spoof_pct']:.0f}% | "
                f"IADG {stats['IADG']['mean_live_score']:.3f}/"
                f"{stats['IADG']['argmax_spoof_pct']:.0f}%"
            )
        lines.append("")

    for key, title in (
        ("phone_genuine_video", "Phone genuine video"),
        ("print_attack_video", "PRINT-like attack video"),
    ):
        v = payload.get(key) or {}
        if v.get("status") != "ok":
            lines.append(f"{title}: {v.get('status')} ({v.get('path')})")
            continue
        lines.append(f"{title} ({v['face_ok_frames']} FACE_OK):")
        lines.append(
            f"  mn3: mean_REAL={v['anti-spoof-mn3']['mean_real']:.4f} "
            f"SPOOF%={v['anti-spoof-mn3']['argmax_spoof_pct']:.1f} "
            f"median_ms={v['anti-spoof-mn3']['median_inference_ms']:.1f}"
        )
        lines.append(
            f"  IADG: mean_live={v['IADG']['mean_live_score']:.4f} "
            f"SPOOF%={v['IADG']['argmax_spoof_pct']:.1f} "
            f"median_ms={v['IADG']['median_inference_ms']:.1f}"
        )
        lines.append("")

    lines += [
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
    parser = argparse.ArgumentParser(description="Benchmark IADG vs mn3")
    parser.add_argument("--phone-video", type=Path, default=PHONE_VIDEO)
    parser.add_argument("--print-video", type=Path, default=PRINT_VIDEO)
    parser.add_argument("--max-face-ok", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()

    print("[INFO] Loading models...", flush=True)
    t0 = time.perf_counter()
    mn3 = LivenessDetector()
    iadg = IadgResearch()
    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    policy = ProvisionalDecisionPolicy()
    print(f"[INFO] Ready in {time.perf_counter() - t0:.1f}s", flush=True)

    camo_runs = []
    for run_dir in HEAD_RUNS:
        print(f"[INFO] Camo run {run_dir.name}...", flush=True)
        camo_runs.append(
            eval_camo_run(run_dir.resolve(), mn3, iadg, face_detector, face_mod, policy)
        )

    if args.skip_videos:
        phone = {"status": "skipped"}
        print_v = {"status": "skipped"}
    else:
        print("[INFO] Phone genuine video...", flush=True)
        phone = eval_video(
            args.phone_video.resolve(),
            "genuine_phone_file",
            mn3,
            iadg,
            face_detector,
            face_mod,
            policy,
            max_face_ok=args.max_face_ok,
            frame_stride=args.frame_stride,
        )
        print("[INFO] PRINT-like video...", flush=True)
        print_v = eval_video(
            args.print_video.resolve(),
            "print_like_attack",
            mn3,
            iadg,
            face_detector,
            face_mod,
            policy,
            max_face_ok=args.max_face_ok,
            frame_stride=args.frame_stride,
        )

    # Summarize answers from available runs
    ok_camo = [r for r in camo_runs if r.get("status") == "ok"]
    camo_spoof_mn3 = [r["overall"]["anti-spoof-mn3"]["argmax_spoof_pct"] for r in ok_camo]
    camo_spoof_iadg = [r["overall"]["IADG"]["argmax_spoof_pct"] for r in ok_camo]
    lat_iadg = [r["overall"]["IADG"]["median_inference_ms"] for r in ok_camo]
    lat_mn3 = [r["overall"]["anti-spoof-mn3"]["median_inference_ms"] for r in ok_camo]

    answers = {
        "genuine_more_reliable": (
            f"Camo SPOOF argmax%: mn3={camo_spoof_mn3}, IADG={camo_spoof_iadg} "
            "(lower better on genuine). Phone SPOOF%: "
            f"mn3={(phone.get('anti-spoof-mn3') or {}).get('argmax_spoof_pct')}, "
            f"IADG={(phone.get('IADG') or {}).get('argmax_spoof_pct')}."
        ),
        "generalize_phone_camo": (
            "Compare phone vs Camo SPOOF%/live means above; "
            "IADG is domain-generalization oriented (OCI2M leave-one-out), "
            "but camera transfer must be judged from these numbers, not assumed."
        ),
        "head_turns": (
            "See by_pose_bucket strong_side_profile / moderate_turn in Camo JSON."
        ),
        "print_rejection": (
            f"PRINT SPOOF%: mn3={(print_v.get('anti-spoof-mn3') or {}).get('argmax_spoof_pct')}, "
            f"IADG={(print_v.get('IADG') or {}).get('argmax_spoof_pct')} "
            f"(status={print_v.get('status')})."
        ),
        "latency": (
            f"Camo median ms mn3={lat_mn3}, IADG={lat_iadg}."
        ),
        "worth_further_or_reject": (
            "Decide from PRINT vs genuine trade-off and latency; "
            "do not replace production without SCREEN evidence."
        ),
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "research_only": True,
        "production_model_unchanged": MN3_NAME,
        "candidate_model": IADG_NAME,
        "iadg_checkpoint": str(iadg.checkpoint_path),
        "iadg_meta": iadg.meta,
        "camo_runs": camo_runs,
        "phone_genuine_video": phone,
        "print_attack_video": print_v,
        "screen_attack": {
            "status": "no_saved_screen_recording",
            "note": "No SCREEN replay video available; not evaluated.",
        },
        "security_note": (
            "Native scores are not calibrated across models. "
            "IADG live_score combines classification and depth heads; "
            "mn3 uses OMZ 2-class probabilities."
        ),
        "answers": answers,
    }

    out_dir = (
        BASE_DIR
        / "diagnostic_head_motion"
        / "20261015T102037Z"
        / "iadg_vs_mn3"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    # Slim JSON without enormous per_sample for secondary run optional - keep full for primary
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = build_table(payload)
    table_path = out_dir / "comparison_table.txt"
    table_path.write_text(table, encoding="utf-8")
    report_path = out_dir / "REPORT.md"
    report_path.write_text(
        "# IADG vs anti-spoof-mn3\n\n```\n" + table + "\n```\n",
        encoding="utf-8",
    )
    print(table)
    print(f"\nJSON: {json_path}")
    missing_any = any(r.get("missing_frames") for r in ok_camo)
    sys.exit(0 if not missing_any else 1)


if __name__ == "__main__":
    main()

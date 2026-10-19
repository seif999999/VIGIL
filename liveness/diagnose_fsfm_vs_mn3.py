"""Research benchmark: FSFM Both_MCIO (CVPR 2025) vs anti-spoof-mn3. Production unchanged."""

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
from research.fsfm_fas import MODEL_NAME as FSFM_NAME, FsfmResearch

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
        "note": "VIGIL provisional policy applies ONLY to mn3 scores",
    }


def _summ_fsfm(rows: list[dict]) -> dict:
    lives = [r["fsfm"]["live_prob"] for r in rows]
    spoof_n = sum(1 for r in rows if r["fsfm"]["predicted_class"] == "SPOOF")
    lat = [r["fsfm"]["inference_ms"] for r in rows]
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
        "decision_rule": (
            "uncalibrated model argmax: class1=REAL/bonafide, class0=SPOOF; "
            "live_prob=softmax[:,1]. NOT a VIGIL policy decision."
        ),
    }


def eval_camo_run(
    run_dir: Path,
    mn3: LivenessDetector,
    fsfm: FsfmResearch,
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
            fsfm_r = fsfm.predict_frame(frame, bbox)
            if fsfm_r is None:
                continue
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
                    "fsfm": {
                        "live_prob": fsfm_r.live_prob,
                        "spoof_prob": fsfm_r.spoof_prob,
                        "predicted_class": fsfm_r.predicted_class.value,
                        "argmax_cls": fsfm_r.argmax_cls,
                        "inference_ms": fsfm_r.inference_ms,
                        "crop_mode": fsfm_r.crop_mode,
                        "device": fsfm_r.device,
                    },
                }
            )

    buckets = sorted({r["pose_bucket"] for r in scored})
    by_bucket = {
        b: {
            "anti-spoof-mn3": _summ_mn3([r for r in scored if r["pose_bucket"] == b]),
            "FSFM": _summ_fsfm([r for r in scored if r["pose_bucket"] == b]),
        }
        for b in buckets
    }
    return {
        "status": "ok",
        "path": str(run_dir),
        "device": data.get("device_resolved"),
        "samples_scored": len(scored),
        "missing_frames": missing,
        "overall": {
            "anti-spoof-mn3": _summ_mn3(scored),
            "FSFM": _summ_fsfm(scored),
        },
        "by_pose_bucket": by_bucket,
        "per_sample": scored,
    }


def eval_video(
    video_path: Path,
    label: str,
    mn3: LivenessDetector,
    fsfm: FsfmResearch,
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
        fsfm_r = fsfm.predict_frame(frame, bbox)
        if fsfm_r is None:
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
                "fsfm": {
                    "live_prob": fsfm_r.live_prob,
                    "spoof_prob": fsfm_r.spoof_prob,
                    "predicted_class": fsfm_r.predicted_class.value,
                    "argmax_cls": fsfm_r.argmax_cls,
                    "inference_ms": fsfm_r.inference_ms,
                    "crop_mode": fsfm_r.crop_mode,
                    "device": fsfm_r.device,
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
        "FSFM": _summ_fsfm(rows),
    }


def build_table(payload: dict) -> str:
    lines = [
        "--- FSFM Both_MCIO (CVPR 2025) vs anti-spoof-mn3 (research only) ---",
        "Production baseline unchanged. Scores are native / uncalibrated across models.",
        f"FSFM checkpoint: {payload['fsfm_checkpoint']}",
        f"FSFM SHA256: {payload['fsfm_meta'].get('sha256')}",
        f"FSFM device: {payload['fsfm_meta'].get('device')} "
        f"peak_cuda_MB={payload.get('fsfm_peak_cuda_mb')}",
        "FSFM score: softmax[:,1]=live_prob; argmax class1=REAL class0=SPOOF",
        "FSFM crop: square bbox * 1.3 expand (official Space), 224 BICUBIC, VF2 mean/std",
        "mn3 crop: production crop_face_bgr expand=1.0; provisional policy on mn3 only",
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
            f"  FSFM: mean_live={o['FSFM']['mean_live_prob']:.4f} "
            f"SPOOF%={o['FSFM']['argmax_spoof_pct']:.1f} "
            f"median_ms={o['FSFM']['median_inference_ms']:.1f}",
            "  By pose (mean | SPOOF%):",
        ]
        for bucket, stats in run["by_pose_bucket"].items():
            lines.append(
                f"    {bucket}: mn3 {stats['anti-spoof-mn3']['mean_real']:.3f}/"
                f"{stats['anti-spoof-mn3']['argmax_spoof_pct']:.0f}% | "
                f"FSFM {stats['FSFM']['mean_live_prob']:.3f}/"
                f"{stats['FSFM']['argmax_spoof_pct']:.0f}%"
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
            f"  FSFM: mean_live={v['FSFM']['mean_live_prob']:.4f} "
            f"SPOOF%={v['FSFM']['argmax_spoof_pct']:.1f} "
            f"median_ms={v['FSFM']['median_inference_ms']:.1f}"
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
    parser = argparse.ArgumentParser(description="Benchmark FSFM vs mn3")
    parser.add_argument("--phone-video", type=Path, default=PHONE_VIDEO)
    parser.add_argument("--print-video", type=Path, default=PRINT_VIDEO)
    parser.add_argument("--max-face-ok", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="cuda|cpu (default auto)")
    args = parser.parse_args()

    print("[INFO] Loading models...", flush=True)
    t0 = time.perf_counter()
    mn3 = LivenessDetector()
    fsfm = FsfmResearch(device=args.device)
    if fsfm.device.type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    policy = ProvisionalDecisionPolicy()
    print(
        f"[INFO] Ready in {time.perf_counter() - t0:.1f}s "
        f"(FSFM on {fsfm.device})",
        flush=True,
    )

    camo_runs = []
    for run_dir in HEAD_RUNS:
        print(f"[INFO] Camo run {run_dir.name}...", flush=True)
        camo_runs.append(
            eval_camo_run(run_dir.resolve(), mn3, fsfm, face_detector, face_mod, policy)
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
            fsfm,
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
            fsfm,
            face_detector,
            face_mod,
            policy,
            max_face_ok=args.max_face_ok,
            frame_stride=args.frame_stride,
        )

    ok_camo = [r for r in camo_runs if r.get("status") == "ok"]
    camo_spoof_mn3 = [r["overall"]["anti-spoof-mn3"]["argmax_spoof_pct"] for r in ok_camo]
    camo_spoof_fsfm = [r["overall"]["FSFM"]["argmax_spoof_pct"] for r in ok_camo]
    camo_live_fsfm = [r["overall"]["FSFM"]["mean_live_prob"] for r in ok_camo]
    camo_real_mn3 = [r["overall"]["anti-spoof-mn3"]["mean_real"] for r in ok_camo]
    lat_fsfm = [r["overall"]["FSFM"]["median_inference_ms"] for r in ok_camo]
    lat_mn3 = [r["overall"]["anti-spoof-mn3"]["median_inference_ms"] for r in ok_camo]

    phone_fsfm_spoof = (phone.get("FSFM") or {}).get("argmax_spoof_pct")
    phone_mn3_spoof = (phone.get("anti-spoof-mn3") or {}).get("argmax_spoof_pct")
    print_fsfm_spoof = (print_v.get("FSFM") or {}).get("argmax_spoof_pct")
    print_mn3_spoof = (print_v.get("anti-spoof-mn3") or {}).get("argmax_spoof_pct")

    improves_genuine = (
        all(s < m for s, m in zip(camo_spoof_fsfm, camo_spoof_mn3))
        and (phone_fsfm_spoof is None or phone_mn3_spoof is None or phone_fsfm_spoof <= phone_mn3_spoof)
    )
    preserves_print = (
        print_fsfm_spoof is not None
        and print_fsfm_spoof >= 95.0
        and (print_mn3_spoof is None or print_fsfm_spoof >= print_mn3_spoof - 5.0)
    )

    answers = {
        "improves_genuine_recognition": (
            f"Camo SPOOF% mn3={camo_spoof_mn3} FSFM={camo_spoof_fsfm}; "
            f"Camo mean mn3_REAL={camo_real_mn3} FSFM_live={camo_live_fsfm}; "
            f"Phone SPOOF% mn3={phone_mn3_spoof} FSFM={phone_fsfm_spoof}. "
            f"Verdict_improves_genuine={improves_genuine}."
        ),
        "preserves_print_rejection": (
            f"PRINT SPOOF% mn3={print_mn3_spoof} FSFM={print_fsfm_spoof} "
            f"(status={print_v.get('status')}). "
            f"Verdict_preserves_print={preserves_print}."
        ),
        "generalize_phone_camo": (
            "Compare phone vs Camo SPOOF%/live means; Both_MCIO is multi-domain FAS, "
            "but camera transfer must be judged from these numbers."
        ),
        "head_pose_robustness": (
            "See by_pose_bucket strong_side_profile / moderate_turn in Camo JSON."
        ),
        "latency_memory": (
            f"Camo median ms mn3={lat_mn3}, FSFM={lat_fsfm}; "
            f"device={fsfm.meta.get('device')}; "
            f"peak_cuda_MB={fsfm.peak_cuda_memory_mb()}."
        ),
        "justifies_compute_cost": (
            "Yes only if genuine improves AND PRINT rejection preserved AND "
            "latency/VRAM acceptable for the deployment target; otherwise reject."
        ),
        "worth_further_or_reject": (
            "Decide from PRINT vs genuine trade-off, camera gap, and cost; "
            "do not replace production without SCREEN evidence."
        ),
    }

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "research_only": True,
        "production_model_unchanged": MN3_NAME,
        "candidate_model": FSFM_NAME,
        "checkpoint_choice_rationale": (
            "Both_MCIO multi-domain FAS ViT-B with 2-class head "
            "(authors' HF Space FAS checkpoint). LOO domain-specific FAS "
            "checkpoints exist but were not used — avoids cherry-picking a "
            "single LOO target domain for our camera eval."
        ),
        "fsfm_checkpoint": str(fsfm.checkpoint_path),
        "fsfm_meta": fsfm.meta,
        "fsfm_peak_cuda_mb": fsfm.peak_cuda_memory_mb(),
        "camo_runs": camo_runs,
        "phone_genuine_video": phone,
        "print_attack_video": print_v,
        "screen_attack": {
            "status": "no_saved_screen_recording",
            "note": "SCREEN marked untested; no replay recording available.",
        },
        "security_note": (
            "Native FSFM live_prob and mn3 REAL scores are uncalibrated across models. "
            "VIGIL provisional / Challenge-1 policy decisions apply ONLY to mn3. "
            "FSFM argmax is a research model prediction, not a production accept/reject."
        ),
        "answers": answers,
        "verdicts": {
            "improves_genuine": improves_genuine,
            "preserves_print": preserves_print,
            "screen_tested": False,
        },
    }

    out_dir = (
        BASE_DIR
        / "diagnostic_head_motion"
        / "20261015T102037Z"
        / "fsfm_vs_mn3"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = build_table(payload)
    table_path = out_dir / "comparison_table.txt"
    table_path.write_text(table, encoding="utf-8")

    # Concise markdown report
    report_lines = [
        "# FSFM Both_MCIO vs anti-spoof-mn3 (research)",
        "",
        "## Checkpoint",
        f"- Hub: `{fsfm.meta['hf_repo']}` / `{fsfm.meta['hf_path']}`",
        f"- SHA256: `{fsfm.meta['sha256']}`",
        f"- License: {fsfm.meta['license']} (research / non-commercial)",
        f"- Device: `{fsfm.meta['device']}` peak CUDA MB: `{payload['fsfm_peak_cuda_mb']}`",
        "",
        "## Scoring (uncalibrated)",
        "| Model | Score | Argmax | Policy |",
        "|---|---|---|---|",
        "| anti-spoof-mn3 | REAL/SPOOF probs | production classes | VIGIL provisional |",
        "| FSFM Both_MCIO | softmax[:,1] live_prob | 1=REAL 0=SPOOF | none (research) |",
        "",
        "## Results",
        "```",
        table,
        "```",
        "",
        "## Limitations",
        "- SCREEN attack: untested (no recording).",
        "- No fine-tuning on VIGIL evaluation media.",
        "- FSFM uses 1.3x square crop; mn3 uses native crop — identical source frames,",
        "  native preprocess each.",
        "- CC BY-NC 4.0: not for commercial production without separate rights.",
        "",
    ]
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(table)
    print(f"\nJSON: {json_path}")
    missing_any = any(r.get("missing_frames") for r in ok_camo)
    sys.exit(0 if not missing_any else 1)


if __name__ == "__main__":
    main()

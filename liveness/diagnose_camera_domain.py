"""Research-only: score phone video vs Camo diagnostic with comparable metrics.

Does not modify production SCRFD, mn3, crops, or thresholds.
Paired causal claims require a same-session recording (see PROTOCOL below).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from crop import crop_face_bgr
from decision_policy import ProvisionalDecisionPolicy, classify_provisional_scores
from detector import LivenessDetector, MODEL_NAME
from diagnose_head_motion_liveness import landmark_yaw_proxy, laplacian_variance
from diagnose_offline_crop_expansion import classify_pose_bucket, resolve_run_path
from face_bridge import load_face_detector_module

BASE_DIR = Path(__file__).resolve().parent
TARGET_FACE_AREA_FRAC = 0.12  # ~12% as in successful phone recording
FACE_AREA_TOL = 0.03  # accept 9%–15% as "comparable to phone"


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def _summarize_rows(rows: list[dict], label: str, source_kind: str) -> dict:
    reals = [r["real_score"] for r in rows]
    spoofs = [r["spoof_score"] for r in rows]
    areas = [r["face_area_frac"] for r in rows]
    laps = [r["laplacian_var"] for r in rows]
    lumas = [r["mean_luma"] for r in rows]
    yaws = [abs(r["yaw_proxy"]) for r in rows if r["yaw_proxy"] is not None]
    prov: dict[str, int] = {}
    argmax: dict[str, int] = {}
    for r in rows:
        prov[r["provisional"]] = prov.get(r["provisional"], 0) + 1
        argmax[r["predicted_class"]] = argmax.get(r["predicted_class"], 0) + 1
    in_band = sum(
        1
        for a in areas
        if abs(a - TARGET_FACE_AREA_FRAC) <= FACE_AREA_TOL
    )
    return {
        "label": label,
        "source_kind": source_kind,
        "n": len(rows),
        "real": {
            "mean": statistics.mean(reals) if reals else None,
            "median": statistics.median(reals) if reals else None,
            "p10": _percentile(reals, 10),
            "p90": _percentile(reals, 90),
            "min": min(reals) if reals else None,
            "max": max(reals) if reals else None,
            "stdev": statistics.stdev(reals) if len(reals) > 1 else None,
        },
        "spoof_mean": statistics.mean(spoofs) if spoofs else None,
        "provisional_counts": prov,
        "model_argmax_counts": argmax,
        "face_area_frac": {
            "mean": statistics.mean(areas) if areas else None,
            "median": statistics.median(areas) if areas else None,
            "target": TARGET_FACE_AREA_FRAC,
            "within_tol_count": in_band,
            "within_tol_pct": 100.0 * in_band / len(areas) if areas else 0.0,
        },
        "laplacian_var": {
            "mean": statistics.mean(laps) if laps else None,
            "median": statistics.median(laps) if laps else None,
        },
        "mean_luma": {
            "mean": statistics.mean(lumas) if lumas else None,
            "median": statistics.median(lumas) if lumas else None,
        },
        "yaw_abs": {
            "mean": statistics.mean(yaws) if yaws else None,
            "max": max(yaws) if yaws else None,
        },
    }


def score_video(
    video_path: Path,
    *,
    max_face_ok: int = 200,
    frame_stride: int = 5,
    phase_hint: str = "mixed",
) -> dict:
    """Offline phone (or any) video through production SCRFD + mn3."""
    if not video_path.is_file():
        return {"status": "missing", "path": str(video_path)}

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    liveness = LivenessDetector()
    policy = ProvisionalDecisionPolicy()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "unreadable", "path": str(video_path)}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        bbox = face.bbox
        crop = crop_face_bgr(frame, bbox)
        if crop is None:
            continue
        live = liveness.predict(crop)
        x1, y1, x2, y2 = bbox
        area = max(x2 - x1, 0) * max(y2 - y1, 0) / max(w * h, 1)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        yaw = landmark_yaw_proxy(face.landmarks)
        rows.append(
            {
                "index": len(rows) + 1,
                "frame_index": idx,
                "real_score": live.real_score,
                "spoof_score": live.spoof_score,
                "predicted_class": live.predicted_class.value,
                "provisional": classify_provisional_scores(
                    live.real_score, live.spoof_score, policy
                ).value,
                "face_area_frac": area,
                "laplacian_var": laplacian_variance(gray),
                "mean_luma": float(np.mean(gray)),
                "yaw_proxy": yaw,
                "crop_w": crop.shape[1],
                "crop_h": crop.shape[0],
            }
        )
    cap.release()

    # Split heuristically: low yaw = frontal-like, high yaw = turn-like
    frontal = [r for r in rows if r["yaw_proxy"] is not None and abs(r["yaw_proxy"]) < 0.035]
    turns = [r for r in rows if r["yaw_proxy"] is not None and abs(r["yaw_proxy"]) >= 0.035]
    return {
        "status": "ok",
        "path": str(video_path),
        "source_kind": "historical_or_file_video",
        "paired_controlled": False,
        "frame_wh": [w, h],
        "model": MODEL_NAME,
        "overall": _summarize_rows(rows, f"video:{video_path.name}", "video_file"),
        "frontal_like_yaw_lt_0.035": _summarize_rows(frontal, "frontal_like", "video_file"),
        "turn_like_yaw_ge_0.035": _summarize_rows(turns, "turn_like", "video_file"),
        "phase_hint": phase_hint,
        "note": (
            "Not a paired Camo session unless recorded same day under PROTOCOL. "
            "Historical phone file is not a controlled camera-domain experiment."
        ),
    }


def score_camo_run(run_dir: Path) -> dict:
    """Re-summarize existing Camo head-motion diagnostic with same metric schema."""
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        return {"status": "missing", "path": str(run_dir)}
    data = json.loads(report_path.read_text(encoding="utf-8"))
    policy = ProvisionalDecisionPolicy()
    samples_b = data["phase_B_head_motion"]["samples"]
    b_max = max(s["index"] for s in samples_b) if samples_b else 0

    def collect(phase_key: str, phase_id: str) -> list[dict]:
        rows: list[dict] = []
        for s in data[phase_key]["samples"]:
            if s.get("real_score") is None or not s.get("bbox"):
                continue
            fw, fh = int(s["frame_w"]), int(s["frame_h"])
            x1, y1, x2, y2 = s["bbox"]
            area = max(x2 - x1, 0) * max(y2 - y1, 0) / max(fw * fh, 1)
            yaw = s.get("landmark_yaw_proxy")
            rows.append(
                {
                    "index": s["index"],
                    "phase": phase_id,
                    "pose_bucket": classify_pose_bucket(
                        phase_id, int(s["index"]), float(yaw or 0), b_max
                    ),
                    "real_score": float(s["real_score"]),
                    "spoof_score": float(s["spoof_score"]),
                    "predicted_class": s["predicted_class"],
                    "provisional": s.get("provisional_outcome")
                    or classify_provisional_scores(
                        float(s["real_score"]),
                        float(s["spoof_score"]),
                        policy,
                    ).value,
                    "face_area_frac": area,
                    "laplacian_var": float(s.get("laplacian_var_crop") or 0),
                    "mean_luma": None,  # not stored live; fill from JPEG if present
                    "yaw_proxy": yaw,
                    "crop_w": s.get("crop_w"),
                    "crop_h": s.get("crop_h"),
                }
            )
        # Optional luma from saved crops
        for r, s in zip(
            rows,
            [x for x in data[phase_key]["samples"] if x.get("real_score") is not None and x.get("bbox")],
            strict=False,
        ):
            crop_rel = s.get("crop_path")
            if not crop_rel:
                continue
            path = resolve_run_path(run_dir, crop_rel)
            if not path.is_file():
                continue
            crop = cv2.imread(str(path))
            if crop is None:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            r["mean_luma"] = float(np.mean(gray))
        # Drop rows without luma into mean using laplacian-only; fill NaN
        for r in rows:
            if r["mean_luma"] is None:
                r["mean_luma"] = 0.0
                r["luma_missing"] = True
        return rows

    rows_a = collect("phase_A_stationary", "A")
    rows_b = collect("phase_B_head_motion", "B")
    return {
        "status": "ok",
        "path": str(run_dir),
        "source_kind": "historical_camo_diagnostic",
        "paired_controlled": False,
        "device": data.get("device_resolved"),
        "model": data.get("liveness_model", MODEL_NAME),
        "phase_A_stationary": _summarize_rows(rows_a, "camo_phase_A", "camo_run"),
        "phase_B_head_motion": _summarize_rows(rows_b, "camo_phase_B", "camo_run"),
        "note": (
            "Historical Camo run - not same-session paired with phone video. "
            "Face area here was ~7%, below phone ~12% target."
        ),
    }


def live_face_area_prep(windows_device: str, duration_s: float = 30.0) -> int:
    """Overlay face/frame % so user can move closer until ~12% before recording."""
    from diagnose_head_motion_liveness import resolve_windows_device_name
    from windows_camera_capture import open_winrt_capture

    device = resolve_windows_device_name(windows_device)
    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector()
    cap = open_winrt_capture(device)
    print(
        f"[PREP] Device={device}. Move closer until face area ≈ {TARGET_FACE_AREA_FRAC*100:.0f}% "
        f"(green band { (TARGET_FACE_AREA_FRAC-FACE_AREA_TOL)*100:.0f}–"
        f"{(TARGET_FACE_AREA_FRAC+FACE_AREA_TOL)*100:.0f}%). Press Q to quit."
    )
    t_end = time.perf_counter() + duration_s
    last_print = 0.0
    try:
        while time.perf_counter() < t_end:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            det = face_detector.detect(frame)
            area = 0.0
            color = (0, 0, 255)
            msg = "NO_FACE"
            if det.decision == face_mod.FaceCountDecision.FACE_OK and det.faces:
                x1, y1, x2, y2 = det.faces[0].bbox
                area = max(x2 - x1, 0) * max(y2 - y1, 0) / (w * h)
                ok_band = abs(area - TARGET_FACE_AREA_FRAC) <= FACE_AREA_TOL
                color = (0, 255, 0) if ok_band else (0, 165, 255)
                msg = f"face={area*100:.1f}%  target={TARGET_FACE_AREA_FRAC*100:.0f}%"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, msg, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.imshow("VIGIL Camo face-area prep (research)", frame)
            if time.perf_counter() - last_print > 1.0:
                print(f"  {msg}")
                last_print = time.perf_counter()
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def build_comparison(phone: dict, camo: dict) -> dict:
    phone_front = phone.get("frontal_like_yaw_lt_0.035") or {}
    camo_a = camo.get("phase_A_stationary") or {}
    phone_o = phone.get("overall") or {}
    return {
        "paired_controlled": False,
        "comparison_valid_for_causality": False,
        "reason": (
            "Sources are historical and not same-session. Use only for hypothesis; "
            "run PROTOCOL for causal camera-domain claim."
        ),
        "face_area_gap": {
            "phone_mean": (phone_o.get("face_area_frac") or {}).get("mean"),
            "camo_phase_A_mean": (camo_a.get("face_area_frac") or {}).get("mean"),
            "target": TARGET_FACE_AREA_FRAC,
        },
        "stationary_or_frontal_real": {
            "phone_frontal_like_mean": (phone_front.get("real") or {}).get("mean"),
            "camo_phase_A_mean": (camo_a.get("real") or {}).get("mean"),
        },
        "provisional": {
            "phone_frontal": phone_front.get("provisional_counts"),
            "camo_phase_A": camo_a.get("provisional_counts"),
        },
    }


PROTOCOL_TEXT = f"""
=== VIGIL camera-domain PROTOCOL (manual; research only) ===

Goal: same person, lighting, distance, pose — compare Camo USB vs native phone video
with face ~{TARGET_FACE_AREA_FRAC*100:.0f}% of frame (as in successful phone recording).

Production model/preprocessing/SCRFD/crops/thresholds: UNCHANGED.

--- Step 1: Phone native video (file, not Camo) ---
1. On the phone, record 1280x720 (or native) selfie / rear video in good light.
2. Phase A: stand still, face camera, ~20s, face filling ~12% of frame
   (move closer until face is clearly larger than a typical webcam selfie distance).
3. Phase B: same distance, slow left/right head turns ~25s.
4. Copy the file locally, e.g.:
   liveness\\camera_domain_runs\\phone_YYYYMMDD.mp4
   (Do not upload; keep local.)

Score it:
  cd "C:\\Users\\smnam\\OneDrive\\Desktop\\VIGIL IMPL\\liveness"
  python diagnose_camera_domain.py --score-video ".\\camera_domain_runs\\phone_YYYYMMDD.mp4"

--- Step 2: Camo prep (match face size) ---
  python diagnose_camera_domain.py --live-prep --windows-device Camo
  Move closer/farther until overlay shows ~12% (green). Then quit (Q).

--- Step 3: Camo diagnostic (same session as phone if possible) ---
  python diagnose_head_motion_liveness.py --windows-device Camo --phase-a-seconds 20 --phase-b-seconds 25
  Keep the face-area band from Step 2 during both phases.

--- Step 4: Compare ---
  python diagnose_camera_domain.py ^
    --score-video ".\\camera_domain_runs\\phone_YYYYMMDD.mp4" ^
    --score-camo-run ".\\diagnostic_head_motion\\<NEW_RUN_TS>" ^
    --out-dir ".\\camera_domain_runs\\compare_<TS>"

Interpretation:
- If Camo at ~12% reaches phone-like mean REAL (>=~0.90) and mostly provisional REAL:
  camera/input conditions are the main lever; keep mn3 and fix capture setup.
- If Camo at ~12% still sits ~0.6 and mostly UNCERTAIN/SPOOF while phone is high:
  model domain robustness is the main gap → evaluate stronger pretrained models
  (still re-test PRINT/SCREEN).
"""


def render_table(payload: dict) -> str:
    lines = [
        "--- Camera-domain metric comparison (research) ---",
        f"Paired controlled: {payload.get('comparison', {}).get('paired_controlled')}",
        f"Causal claim allowed: {payload.get('comparison', {}).get('comparison_valid_for_causality')}",
        "",
    ]
    phone = payload.get("phone_video") or {}
    camo = payload.get("camo_run") or {}
    if phone.get("status") == "ok":
        o = phone["overall"]
        f = phone["frontal_like_yaw_lt_0.035"]
        lines += [
            f"PHONE FILE ({phone['path']}):",
            f"  overall mean REAL={o['real']['mean']:.4f}  n={o['n']}  "
            f"face_area={o['face_area_frac']['mean']:.3f}  "
            f"lap={o['laplacian_var']['mean']:.1f}  luma={o['mean_luma']['mean']:.1f}",
            f"  provisional={o['provisional_counts']}  argmax={o['model_argmax_counts']}",
            f"  frontal-like mean REAL={f['real']['mean'] if f.get('real', {}).get('mean') is not None else None}  "
            f"n={f['n']}  provisional={f.get('provisional_counts')}",
            f"  note: {phone.get('note')}",
            "",
        ]
    if camo.get("status") == "ok":
        a = camo["phase_A_stationary"]
        b = camo["phase_B_head_motion"]
        lines += [
            f"CAMO RUN ({camo['path']}):",
            f"  Phase A mean REAL={a['real']['mean']:.4f}  n={a['n']}  "
            f"face_area={a['face_area_frac']['mean']:.3f}  "
            f"lap={a['laplacian_var']['mean']:.1f}  luma={a['mean_luma']['mean']:.1f}",
            f"  provisional={a['provisional_counts']}  argmax={a['model_argmax_counts']}",
            f"  Phase B mean REAL={b['real']['mean']:.4f}  n={b['n']}  "
            f"face_area={b['face_area_frac']['mean']:.3f}",
            f"  provisional={b['provisional_counts']}  argmax={b['model_argmax_counts']}",
            f"  note: {camo.get('note')}",
            "",
        ]
    if payload.get("comparison"):
        c = payload["comparison"]
        lines += [
            "Gap (historical, not causal):",
            f"  face_area phone vs camo A: {c['face_area_gap']}",
            f"  REAL phone frontal vs camo A: {c['stationary_or_frontal_real']}",
            f"  {c['reason']}",
            "",
        ]
    lines.append("Next: run PROTOCOL (printed with --print-protocol) for a paired session.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Camera-domain research comparison")
    parser.add_argument(
        "--score-video",
        type=Path,
        default=None,
        help="Native phone (or other) video file",
    )
    parser.add_argument(
        "--score-camo-run",
        type=Path,
        default=None,
        help="Existing diagnose_head_motion run directory",
    )
    parser.add_argument("--live-prep", action="store_true")
    parser.add_argument("--windows-device", type=str, default="Camo")
    parser.add_argument("--prep-seconds", type=float, default=45.0)
    parser.add_argument("--max-face-ok", type=int, default=200)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--print-protocol", action="store_true")
    parser.add_argument(
        "--use-historical-defaults",
        action="store_true",
        help="Score historical phone video + Camo 20261015T102037Z (not paired)",
    )
    args = parser.parse_args()

    if args.print_protocol:
        print(PROTOCOL_TEXT)
        if not (
            args.live_prep
            or args.score_video
            or args.score_camo_run
            or args.use_historical_defaults
        ):
            return

    if args.live_prep:
        sys.exit(live_face_area_prep(args.windows_device, args.prep_seconds))

    phone_path = args.score_video
    camo_path = args.score_camo_run
    if args.use_historical_defaults:
        phone_path = phone_path or (BASE_DIR.parent / "face_detection" / "test video.mp4")
        camo_path = camo_path or (
            BASE_DIR / "diagnostic_head_motion" / "20261015T102037Z"
        )

    if phone_path is None and camo_path is None:
        print(PROTOCOL_TEXT)
        print(
            "[ERROR] Provide --score-video / --score-camo-run / "
            "--use-historical-defaults / --live-prep / --print-protocol",
            file=sys.stderr,
        )
        sys.exit(2)

    out_dir = args.out_dir
    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = BASE_DIR / "camera_domain_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "research_only": True,
        "production_unchanged": True,
        "target_face_area_frac": TARGET_FACE_AREA_FRAC,
    }
    if phone_path is not None:
        print(f"[INFO] Scoring video: {phone_path}", flush=True)
        payload["phone_video"] = score_video(
            phone_path.resolve(),
            max_face_ok=args.max_face_ok,
            frame_stride=args.frame_stride,
        )
    if camo_path is not None:
        print(f"[INFO] Scoring Camo run: {camo_path}", flush=True)
        payload["camo_run"] = score_camo_run(camo_path.resolve())
    if payload.get("phone_video", {}).get("status") == "ok" and payload.get(
        "camo_run", {}
    ).get("status") == "ok":
        payload["comparison"] = build_comparison(
            payload["phone_video"], payload["camo_run"]
        )

    json_path = out_dir / "camera_domain_report.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table = render_table(payload)
    table_path = out_dir / "camera_domain_table.txt"
    table_path.write_text(table + "\n\n" + PROTOCOL_TEXT, encoding="utf-8")
    protocol_path = out_dir / "PROTOCOL.txt"
    protocol_path.write_text(PROTOCOL_TEXT, encoding="utf-8")
    print(table)
    print(f"\nJSON: {json_path}")
    print(f"Table+protocol: {table_path}")


if __name__ == "__main__":
    main()

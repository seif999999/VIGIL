"""Offline: compare 1.0x / 1.2x / 1.4x SCRFD bbox crops on saved head-motion frames."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from crop import crop_face_bgr
from detector import LivenessDetector, MODEL_NAME

BASE_DIR = Path(__file__).resolve().parent
EXPANSION_FACTORS = (1.0, 1.2, 1.4)
# Saved crops are lossy JPEG; full frames are JPEG too. Exact bitwise replay is not expected.
REPRO_SAVED_CROP_TOLERANCE = 0.025
REPRO_FRAME_BBOX_TOLERANCE = 0.25


@dataclass
class ScoredSample:
    phase: str
    index: int
    pose_bucket: str
    yaw_proxy: float
    recorded_real: float
    recorded_spoof: float
    recorded_class: str
    by_factor: dict[float, dict]


def resolve_run_path(run_dir: Path, rel_path: str) -> Path:
    return (run_dir / Path(rel_path.replace("\\", "/"))).resolve()


def classify_pose_bucket(phase: str, index: int, yaw: float, b_index_max: int) -> str:
    ay = abs(yaw)
    if phase == "A":
        return "stationary_frontal"
    # Phase B
    if index > b_index_max * 0.55 and ay < 0.035:
        return "returned_to_front"
    if ay >= 0.12:
        return "strong_side_profile"
    if ay >= 0.035:
        return "moderate_turn"
    return "returned_to_front" if index > b_index_max * 0.5 else "moderate_turn"


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


def summarize_bucket(rows: list[ScoredSample], factor: float) -> dict:
    reals = [r.by_factor[factor]["real_score"] for r in rows]
    spoofs = [r.by_factor[factor]["spoof_score"] for r in rows]
    spoof_preds = sum(
        1 for r in rows if r.by_factor[factor]["predicted_class"] == "SPOOF"
    )
    return {
        "n": len(rows),
        "mean_real": statistics.mean(reals) if reals else None,
        "median_real": statistics.median(reals) if reals else None,
        "p10_real": _percentile(reals, 10) if reals else None,
        "p90_real": _percentile(reals, 90) if reals else None,
        "mean_spoof": statistics.mean(spoofs) if spoofs else None,
        "model_argmax_spoof_count": spoof_preds,
        "model_argmax_spoof_pct": (100.0 * spoof_preds / len(rows)) if rows else 0.0,
    }


def build_representative_panel(
    run_dir: Path,
    out_dir: Path,
    picks: list[tuple[str, ScoredSample]],
) -> Path | None:
    tiles: list[np.ndarray] = []
    thumb_h = 100
    for label, row in picks:
        frame_path = resolve_run_path(run_dir, row.by_factor[1.0]["frame_path"])
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        bbox = tuple(row.by_factor[1.0]["bbox"])
        strip_parts: list[np.ndarray] = []
        for fac in EXPANSION_FACTORS:
            crop = crop_face_bgr(frame, bbox, expand_factor=fac)
            if crop is None:
                continue
            sc = row.by_factor[fac]
            scale = thumb_h / max(crop.shape[0], 1)
            thumb = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), thumb_h),
                interpolation=cv2.INTER_AREA,
            )
            cv2.putText(
                thumb,
                f"{fac}x R={sc['real_score']:.2f}",
                (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                (0, 255, 0),
                1,
            )
            strip_parts.append(thumb)
        if not strip_parts:
            continue
        strip = np.hstack(strip_parts)
        banner = np.zeros((22, strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            banner,
            f"{label} #{row.index} yaw={row.yaw_proxy:+.2f}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 200, 0),
            1,
        )
        tiles.append(np.vstack([banner, strip]))
    if not tiles:
        return None
    panel = tiles[0]
    for t in tiles[1:]:
        w = max(panel.shape[1], t.shape[1])
        if panel.shape[1] < w:
            panel = np.hstack(
                [panel, np.zeros((panel.shape[0], w - panel.shape[1], 3), np.uint8)]
            )
        if t.shape[1] < w:
            t = np.hstack([t, np.zeros((t.shape[0], w - t.shape[1], 3), np.uint8)])
        panel = np.vstack([panel, np.zeros((6, w, 3), np.uint8), t])
    out_path = out_dir / "representative_crops_panel.png"
    cv2.imwrite(str(out_path), panel)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline crop expansion study on saved head-motion diagnostic",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=BASE_DIR / "diagnostic_head_motion" / "20261015T102037Z",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        print(f"[ERROR] Missing {report_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    samples_a = data["phase_A_stationary"]["samples"]
    samples_b = data["phase_B_head_motion"]["samples"]
    b_max_index = max(s["index"] for s in samples_b) if samples_b else 0

    liveness = LivenessDetector()
    scored: list[ScoredSample] = []
    repro_frame_bbox: list[dict] = []
    repro_saved_crop: list[dict] = []
    missing_frames: list[str] = []

    for phase_key, phase_id in (
        ("phase_A_stationary", "A"),
        ("phase_B_head_motion", "B"),
    ):
        for s in data[phase_key]["samples"]:
            if s.get("real_score") is None or not s.get("bbox"):
                continue
            frame_rel = s.get("frame_path")
            if not frame_rel:
                continue
            frame_path = resolve_run_path(run_dir, frame_rel)
            if not frame_path.is_file():
                missing_frames.append(str(frame_rel))
                continue
            frame = cv2.imread(str(frame_path))
            if frame is None:
                missing_frames.append(str(frame_rel))
                continue
            bbox = tuple(s["bbox"])
            yaw = float(s["landmark_yaw_proxy"])
            bucket = classify_pose_bucket(phase_id, int(s["index"]), yaw, b_max_index)
            by_factor: dict[float, dict] = {}
            for fac in EXPANSION_FACTORS:
                crop = crop_face_bgr(frame, bbox, expand_factor=fac)
                if crop is None:
                    continue
                live = liveness.predict(crop)
                by_factor[fac] = {
                    "real_score": live.real_score,
                    "spoof_score": live.spoof_score,
                    "predicted_class": live.predicted_class.value,
                    "crop_w": int(crop.shape[1]),
                    "crop_h": int(crop.shape[0]),
                    "bbox": list(bbox),
                    "frame_path": frame_rel,
                }
            if 1.0 not in by_factor:
                continue
            diff_frame = abs(by_factor[1.0]["real_score"] - float(s["real_score"]))
            repro_frame_bbox.append(
                {
                    "phase": phase_id,
                    "index": s["index"],
                    "recorded_real": s["real_score"],
                    "recomputed_real": by_factor[1.0]["real_score"],
                    "abs_diff": diff_frame,
                }
            )
            crop_rel = s.get("crop_path")
            if crop_rel:
                crop_path = resolve_run_path(run_dir, crop_rel)
                saved = cv2.imread(str(crop_path))
                if saved is not None:
                    live_saved = liveness.predict(saved)
                    diff_crop = abs(live_saved.real_score - float(s["real_score"]))
                    repro_saved_crop.append(
                        {
                            "phase": phase_id,
                            "index": s["index"],
                            "recorded_real": s["real_score"],
                            "from_saved_crop_real": live_saved.real_score,
                            "abs_diff": diff_crop,
                        }
                    )
            scored.append(
                ScoredSample(
                    phase=phase_id,
                    index=int(s["index"]),
                    pose_bucket=bucket,
                    yaw_proxy=yaw,
                    recorded_real=float(s["real_score"]),
                    recorded_spoof=float(s["spoof_score"]),
                    recorded_class=s["predicted_class"],
                    by_factor=by_factor,
                )
            )

    buckets = sorted({r.pose_bucket for r in scored})
    table: dict[str, dict[str, dict]] = {}
    for bucket in buckets:
        rows = [r for r in scored if r.pose_bucket == bucket]
        table[bucket] = {
            str(fac): summarize_bucket(rows, fac) for fac in EXPANSION_FACTORS
        }

    overall = {str(fac): summarize_bucket(scored, fac) for fac in EXPANSION_FACTORS}

    out_dir = run_dir / "crop_expansion_offline"
    out_dir.mkdir(parents=True, exist_ok=True)

    picks: list[tuple[str, ScoredSample]] = []
    for bucket in buckets:
        rows = [r for r in scored if r.pose_bucket == bucket]
        if not rows:
            continue
        if bucket == "strong_side_profile":
            pick = min(rows, key=lambda r: r.by_factor[1.0]["real_score"])
        elif bucket == "stationary_frontal":
            pick = max(rows, key=lambda r: r.by_factor[1.0]["real_score"])
        else:
            pick = rows[len(rows) // 2]
        picks.append((bucket, pick))
    panel_path = build_representative_panel(run_dir, out_dir, picks)

    payload = {
        "run_dir": str(run_dir),
        "model": MODEL_NAME,
        "expansion_factors": list(EXPANSION_FACTORS),
        "repro_saved_crop_tolerance": REPRO_SAVED_CROP_TOLERANCE,
        "repro_frame_bbox_tolerance": REPRO_FRAME_BBOX_TOLERANCE,
        "samples_scored": len(scored),
        "repro_1p0x_from_saved_crop": {
            "n": len(repro_saved_crop),
            "within_tolerance": sum(
                1
                for r in repro_saved_crop
                if r["abs_diff"] <= REPRO_SAVED_CROP_TOLERANCE
            ),
            "max_abs_diff": max((r["abs_diff"] for r in repro_saved_crop), default=0.0),
            "mean_abs_diff": statistics.mean(
                [r["abs_diff"] for r in repro_saved_crop]
            )
            if repro_saved_crop
            else None,
            "failures": [
                r
                for r in repro_saved_crop
                if r["abs_diff"] > REPRO_SAVED_CROP_TOLERANCE
            ],
        },
        "repro_1p0x_from_frame_bbox": {
            "n": len(repro_frame_bbox),
            "within_tolerance": sum(
                1
                for r in repro_frame_bbox
                if r["abs_diff"] <= REPRO_FRAME_BBOX_TOLERANCE
            ),
            "max_abs_diff": max((r["abs_diff"] for r in repro_frame_bbox), default=0.0),
            "mean_abs_diff": statistics.mean(
                [r["abs_diff"] for r in repro_frame_bbox]
            )
            if repro_frame_bbox
            else None,
        },
        "missing_frames": missing_frames,
        "overall_by_factor": overall,
        "by_pose_bucket": table,
        "security_note": (
            "Genuine-person crop expansion gains do not validate PRINT/SCREEN "
            "attack resistance; reevaluate presentation-attack metrics before adoption."
        ),
    }
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "--- Offline crop expansion (diagnostic only) ---",
        f"Run: {run_dir}",
        f"Samples: {len(scored)}",
        f"1.0x repro (saved crop JPG, tol={REPRO_SAVED_CROP_TOLERANCE}): "
        f"{payload['repro_1p0x_from_saved_crop']['within_tolerance']}/"
        f"{payload['repro_1p0x_from_saved_crop']['n']} within; "
        f"mean |diff|={payload['repro_1p0x_from_saved_crop']['mean_abs_diff']:.4f}",
        f"1.0x repro (frame JPG + bbox, tol={REPRO_FRAME_BBOX_TOLERANCE}): "
        f"{payload['repro_1p0x_from_frame_bbox']['within_tolerance']}/"
        f"{payload['repro_1p0x_from_frame_bbox']['n']} within; "
        f"mean |diff|={payload['repro_1p0x_from_frame_bbox']['mean_abs_diff']:.4f}",
        f"Missing frames: {len(missing_frames)}",
        "",
        "Overall mean REAL / SPOOF argmax count:",
    ]
    for fac in EXPANSION_FACTORS:
        o = overall[str(fac)]
        lines.append(
            f"  {fac}x: mean_REAL={o['mean_real']:.4f} "
            f"median_REAL={o['median_real']:.4f} "
            f"SPOOF_preds={o['model_argmax_spoof_count']}/{o['n']} "
            f"({o['model_argmax_spoof_pct']:.1f}%)"
        )
    lines.append("")
    lines.append("| Pose bucket | 1.0x mean REAL | 1.2x | 1.4x | "
                 "1.0x SPOOF% | 1.2x | 1.4x |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for bucket in buckets:
        c0 = table[bucket]["1.0"]
        c12 = table[bucket]["1.2"]
        c14 = table[bucket]["1.4"]
        lines.append(
            f"| {bucket} | {c0['mean_real']:.3f} | {c12['mean_real']:.3f} | "
            f"{c14['mean_real']:.3f} | {c0['model_argmax_spoof_pct']:.0f}% | "
            f"{c12['model_argmax_spoof_pct']:.0f}% | {c14['model_argmax_spoof_pct']:.0f}% |"
        )
    lines.append("")
    lines.append(payload["security_note"])
    if panel_path:
        lines.append(f"Panel: {panel_path}")
    lines.append(f"JSON: {json_path}")

    summary_path = out_dir / "comparison_table.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if not missing_frames else 1)


if __name__ == "__main__":
    main()

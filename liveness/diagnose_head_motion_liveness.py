"""Two-phase diagnostic: stationary vs head motion → SCRFD crop → anti-spoof-mn3 scores."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from crop import crop_face_bgr
from decision_policy import ProvisionalDecisionPolicy, classify_provisional_scores
from detector import LivenessDetector, MODEL_NAME
from face_bridge import load_face_detector_module

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BASE_DIR / "diagnostic_head_motion"


def resolve_windows_device_name(query: str) -> str:
    """Match MediaFrameSourceGroup display name (substring, case-insensitive)."""

    async def _find() -> str:
        from winrt.windows.media.capture.frames import MediaFrameSourceGroup

        groups = await MediaFrameSourceGroup.find_all_async()
        names = [g.display_name for g in groups]
        q = query.lower()
        hits = [n for n in names if q in n.lower()]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise RuntimeError(
                f"No Windows camera matching {query!r}. Available: {names}"
            )
        raise RuntimeError(
            f"Ambiguous camera {query!r}. Matches: {hits}. Use exact display name."
        )

    return asyncio.run(_find())


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def landmark_yaw_proxy(landmarks: np.ndarray) -> float | None:
    """Approximate in-plane yaw from SCRFD 5-point layout (InsightFace order)."""
    if landmarks is None or landmarks.shape != (5, 2):
        return None
    le, re, nose = landmarks[0], landmarks[1], landmarks[2]
    eye_mid = (le + re) * 0.5
    inter_eye = float(np.linalg.norm(re - le))
    if inter_eye < 1e-3:
        return None
    return float((nose[0] - eye_mid[0]) / inter_eye)


@dataclass
class SampleRecord:
    phase: str
    index: int
    timestamp_s: float
    frame_w: int
    frame_h: int
    face_decision: str
    det_confidence: float
    bbox: list[float]
    crop_w: int
    crop_h: int
    crop_aspect: float
    laplacian_var_crop: float
    landmark_yaw_proxy: float | None
    real_score: float | None
    spoof_score: float | None
    predicted_class: str | None
    provisional_outcome: str | None
    frame_path: str
    bbox_path: str | None
    crop_path: str | None


def _draw_bbox_frame(frame_bgr: np.ndarray, face, decision: str) -> np.ndarray:
    out = frame_bgr.copy()
    x1, y1, x2, y2 = map(int, face.bbox)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for px, py in face.landmarks:
        cv2.circle(out, (int(px), int(py)), 2, (0, 255, 255), -1)
    cv2.putText(
        out,
        f"{decision} conf={face.confidence:.2f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return out


def _rel_to_run(run_dir: Path, absolute: Path) -> str:
    return absolute.relative_to(run_dir).as_posix()


def resolve_asset_path(run_dir: Path, rel_path: str | None) -> Path | None:
    if not rel_path:
        return None
    candidate = (run_dir / Path(rel_path.replace("\\", "/"))).resolve()
    run_root = run_dir.resolve()
    if run_root not in candidate.parents and candidate != run_root:
        return None
    return candidate if candidate.is_file() else None


def _scored_with_yaw(records: list[SampleRecord]) -> list[SampleRecord]:
    return [
        r
        for r in records
        if r.crop_path and r.landmark_yaw_proxy is not None and r.real_score is not None
    ]


def select_contact_sheet_rows(
    phase_a: list[SampleRecord],
    phase_b: list[SampleRecord],
) -> list[tuple[str, list[SampleRecord]]]:
    rows: list[tuple[str, list[SampleRecord]]] = []
    a_scored = _scored_with_yaw(phase_a)
    if a_scored:
        stationary = sorted(a_scored, key=lambda r: abs(r.landmark_yaw_proxy))[:3]
        rows.append(("Phase A — stationary (front)", stationary))

    b_scored = _scored_with_yaw(phase_b)
    if b_scored:
        turn_left = min(b_scored, key=lambda r: r.landmark_yaw_proxy)
        turn_right = max(b_scored, key=lambda r: r.landmark_yaw_proxy)
        by_index = sorted(b_scored, key=lambda r: r.index)
        second_half = by_index[len(by_index) // 2 :]
        return_front = min(
            second_half or by_index,
            key=lambda r: abs(r.landmark_yaw_proxy),
        )
        rows.append(("Phase B — turn left", [turn_left]))
        rows.append(("Phase B — turn right", [turn_right]))
        rows.append(("Phase B — return to front", [return_front]))
    return rows


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


def _phase_stats(records: list[SampleRecord]) -> dict:
    scored = [r for r in records if r.real_score is not None]
    reals = [r.real_score for r in scored]
    spoofs = [r.spoof_score for r in scored]
    laps = [r.laplacian_var_crop for r in scored]
    yaws = [r.landmark_yaw_proxy for r in scored if r.landmark_yaw_proxy is not None]
    bw = [r.crop_w for r in scored]
    bh = [r.crop_h for r in scored]
    policy = ProvisionalDecisionPolicy()
    prov = [
        classify_provisional_scores(r.real_score, r.spoof_score, policy).value
        for r in scored
        if r.real_score is not None and r.spoof_score is not None
    ]
    prov_counts = {k: prov.count(k) for k in sorted(set(prov))}
    return {
        "samples_total": len(records),
        "face_ok_scored": len(scored),
        "real_score_mean": statistics.mean(reals) if reals else None,
        "real_score_median": statistics.median(reals) if reals else None,
        "real_score_p10": _percentile(reals, 10) if reals else None,
        "real_score_p90": _percentile(reals, 90) if reals else None,
        "spoof_score_mean": statistics.mean(spoofs) if spoofs else None,
        "spoof_score_median": statistics.median(spoofs) if spoofs else None,
        "laplacian_var_mean": statistics.mean(laps) if laps else None,
        "laplacian_var_median": statistics.median(laps) if laps else None,
        "crop_width_mean": statistics.mean(bw) if bw else None,
        "crop_height_mean": statistics.mean(bh) if bh else None,
        "yaw_proxy_mean": statistics.mean(yaws) if yaws else None,
        "yaw_proxy_stdev": statistics.pstdev(yaws) if len(yaws) > 1 else None,
        "yaw_proxy_abs_max": max(abs(y) for y in yaws) if yaws else None,
        "provisional_outcome_counts": prov_counts,
        "model_argmax_real": sum(1 for r in scored if r.predicted_class == "REAL"),
        "model_argmax_spoof": sum(1 for r in scored if r.predicted_class == "SPOOF"),
    }


def build_contact_sheet(
    phase_a: list[SampleRecord],
    phase_b: list[SampleRecord],
    out_path: Path,
    run_dir: Path,
) -> tuple[bool, list[str]]:
    """Build pose-labelled contact sheet; paths resolved under run_dir."""
    missing: list[str] = []
    row_specs = select_contact_sheet_rows(phase_a, phase_b)
    image_rows: list[np.ndarray] = []
    thumb_h = 128

    for label, recs in row_specs:
        tiles: list[np.ndarray] = []
        for r in recs:
            crop_file = resolve_asset_path(run_dir, r.crop_path)
            if crop_file is None:
                missing.append(f"{label}: missing crop {r.crop_path!r}")
                continue
            img = cv2.imread(str(crop_file))
            if img is None:
                missing.append(f"{label}: unreadable {crop_file}")
                continue
            scale = thumb_h / max(img.shape[0], 1)
            thumb = cv2.resize(
                img,
                (int(img.shape[1] * scale), thumb_h),
                interpolation=cv2.INTER_AREA,
            )
            real_txt = (
                f"REAL={r.real_score:.3f}" if r.real_score is not None else "REAL=n/a"
            )
            tag = f"#{r.index} {real_txt}"
            cv2.putText(
                thumb,
                tag,
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 0),
                1,
            )
            if r.landmark_yaw_proxy is not None:
                cv2.putText(
                    thumb,
                    f"yaw={r.landmark_yaw_proxy:+.2f}",
                    (4, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 0),
                    1,
                )
            tiles.append(thumb)
        if not tiles:
            continue
        row = np.hstack(tiles)
        banner = np.zeros((32, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            banner,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            1,
        )
        image_rows.append(np.vstack([banner, row]))

    if not image_rows:
        return False, missing

    gap = np.zeros((8, max(r.shape[1] for r in image_rows), 3), dtype=np.uint8)
    sheet = image_rows[0]
    for r in image_rows[1:]:
        w = max(sheet.shape[1], r.shape[1])
        if sheet.shape[1] < w:
            pad = np.zeros((sheet.shape[0], w - sheet.shape[1], 3), dtype=np.uint8)
            sheet = np.hstack([sheet, pad])
        if r.shape[1] < w:
            pad = np.zeros((r.shape[0], w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        sheet = np.vstack([sheet, gap[:, :w], r])
    if not cv2.imwrite(str(out_path), sheet):
        missing.append(f"failed to write {out_path}")
        return False, missing
    return True, missing


def regenerate_contact_sheet_from_run(run_dir: Path) -> int:
    report_path = run_dir / "report.json"
    if not report_path.is_file():
        print(f"[ERROR] Missing {report_path}", file=sys.stderr)
        return 1
    data = json.loads(report_path.read_text(encoding="utf-8"))
    records_a = [
        SampleRecord(**s) for s in data["phase_A_stationary"]["samples"]
    ]
    records_b = [
        SampleRecord(**s) for s in data["phase_B_head_motion"]["samples"]
    ]
    out_path = run_dir / "contact_sheet.png"
    ok, missing = build_contact_sheet(records_a, records_b, out_path, run_dir)
    if missing:
        print("Contact sheet warnings:")
        for line in missing:
            print(f"  - {line}")
    if ok:
        print(f"Wrote {out_path}")
        return 0
    print("[ERROR] Contact sheet not created.", file=sys.stderr)
    return 1


def run_phase(
    *,
    phase_id: str,
    phase_label: str,
    duration_s: float,
    sample_interval_s: float,
    cap,
    face_detector,
    liveness: LivenessDetector,
    out_samples: Path,
    policy: ProvisionalDecisionPolicy,
) -> list[SampleRecord]:
    print(f"\n=== {phase_label} ({duration_s:.0f}s) ===")
    print(phase_label)
    records: list[SampleRecord] = []
    t_end = time.perf_counter() + duration_s
    next_sample = time.perf_counter()
    idx = 0

    while time.perf_counter() < t_end:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        now = time.perf_counter()
        if now < next_sample:
            continue
        next_sample = now + sample_interval_s
        idx += 1
        h, w = frame.shape[:2]
        det = face_detector.detect(frame)
        rec = SampleRecord(
            phase=phase_id,
            index=idx,
            timestamp_s=now,
            frame_w=w,
            frame_h=h,
            face_decision=det.decision.value,
            det_confidence=0.0,
            bbox=[],
            crop_w=0,
            crop_h=0,
            crop_aspect=0.0,
            laplacian_var_crop=0.0,
            landmark_yaw_proxy=None,
            real_score=None,
            spoof_score=None,
            predicted_class=None,
            provisional_outcome=None,
            frame_path="",
            bbox_path=None,
            crop_path=None,
        )
        stem = f"{idx:04d}"
        frame_path = out_samples / f"{stem}_frame.jpg"
        cv2.imwrite(str(frame_path), frame)
        rec.frame_path = _rel_to_run(out_samples.parent.parent, frame_path)

        if det.decision.value == "FACE_OK" and det.faces:
            face = det.faces[0]
            rec.det_confidence = face.confidence
            rec.bbox = list(face.bbox)
            crop = crop_face_bgr(frame, face.bbox)
            yaw = landmark_yaw_proxy(face.landmarks)
            rec.landmark_yaw_proxy = yaw
            bbox_img = _draw_bbox_frame(frame, face, det.decision.value)
            bbox_path = out_samples / f"{stem}_bbox.jpg"
            cv2.imwrite(str(bbox_path), bbox_img)
            rec.bbox_path = _rel_to_run(out_samples.parent.parent, bbox_path)
            if crop is not None:
                ch, cw = crop.shape[:2]
                rec.crop_w = cw
                rec.crop_h = ch
                rec.crop_aspect = cw / ch if ch else 0.0
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                rec.laplacian_var_crop = laplacian_variance(gray)
                crop_path = out_samples / f"{stem}_crop.jpg"
                cv2.imwrite(str(crop_path), crop)
                rec.crop_path = _rel_to_run(out_samples.parent.parent, crop_path)
                live = liveness.predict(crop)
                rec.real_score = live.real_score
                rec.spoof_score = live.spoof_score
                rec.predicted_class = live.predicted_class.value
                rec.provisional_outcome = classify_provisional_scores(
                    live.real_score, live.spoof_score, policy
                ).value
        records.append(rec)
        if rec.real_score is not None:
            print(
                f"  [{stem}] real={rec.real_score:.4f} spoof={rec.spoof_score:.4f} "
                f"yaw={rec.landmark_yaw_proxy} lap={rec.laplacian_var_crop:.1f} "
                f"crop={rec.crop_w}x{rec.crop_h}"
            )
        else:
            print(f"  [{stem}] {det.decision.value}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stationary vs head-motion anti-spoof-mn3 diagnostic (Camo / WinRT)",
    )
    parser.add_argument(
        "--windows-device",
        type=str,
        default=None,
        help='Device substring or exact name, e.g. "Camo"',
    )
    parser.add_argument(
        "--regenerate-contact-sheet",
        type=Path,
        metavar="RUN_DIR",
        default=None,
        help="Rebuild contact_sheet.png from an existing diagnostic run folder",
    )
    parser.add_argument("--phase-a-seconds", type=float, default=20.0)
    parser.add_argument("--phase-b-seconds", type=float, default=25.0)
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.35,
        help="Seconds between saved samples within each phase",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pause-between-phases",
        type=float,
        default=5.0,
        help="Seconds between phase A and B (time to reposition)",
    )
    args = parser.parse_args()

    if args.regenerate_contact_sheet is not None:
        sys.exit(regenerate_contact_sheet_from_run(args.regenerate_contact_sheet))

    if not args.windows_device:
        print(
            "[ERROR] Provide --windows-device or --regenerate-contact-sheet",
            file=sys.stderr,
        )
        sys.exit(2)

    device_name = resolve_windows_device_name(args.windows_device)
    print(f"[INFO] Resolved device: {device_name}")

    from windows_camera_capture import open_winrt_capture, print_winrt_capture_info

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / ts
    samples_a = run_dir / "phase_A_stationary" / "samples"
    samples_b = run_dir / "phase_B_head_motion" / "samples"
    samples_a.mkdir(parents=True, exist_ok=True)
    samples_b.mkdir(parents=True, exist_ok=True)

    face_mod = load_face_detector_module()
    face_detector = face_mod.FaceDetector(det_thresh=args.threshold)
    liveness = LivenessDetector()
    policy = ProvisionalDecisionPolicy()

    cap = open_winrt_capture(device_name)
    print_winrt_capture_info(cap.capture_info)
    capture_info = {
        "width": cap.capture_info.actual_width,
        "height": cap.capture_info.actual_height,
        "device_name": cap.capture_info.device_name,
    }

    try:
        records_a = run_phase(
            phase_id="A",
            phase_label="Phase A — genuine person STATIONARY, face camera",
            duration_s=args.phase_a_seconds,
            sample_interval_s=args.sample_interval,
            cap=cap,
            face_detector=face_detector,
            liveness=liveness,
            out_samples=samples_a,
            policy=policy,
        )
        if args.pause_between_phases > 0:
            print(
                f"\n[PROMPT] Prepare for Phase B in {args.pause_between_phases:.0f}s: "
                "slowly turn head left and right."
            )
            time.sleep(args.pause_between_phases)
        records_b = run_phase(
            phase_id="B",
            phase_label="Phase B — same person, SLOW head turn left/right",
            duration_s=args.phase_b_seconds,
            sample_interval_s=args.sample_interval,
            cap=cap,
            face_detector=face_detector,
            liveness=liveness,
            out_samples=samples_b,
            policy=policy,
        )
    finally:
        cap.release()

    stats_a = _phase_stats(records_a)
    stats_b = _phase_stats(records_b)
    contact_path = run_dir / "contact_sheet.png"
    ok_sheet, sheet_missing = build_contact_sheet(
        records_a, records_b, contact_path, run_dir
    )
    if sheet_missing:
        print("Contact sheet warnings:")
        for line in sheet_missing:
            print(f"  - {line}")
    if not ok_sheet:
        print("[WARN] contact_sheet.png was not created.", file=sys.stderr)

    report = {
        "timestamp_utc": ts,
        "device_resolved": device_name,
        "device_query": args.windows_device,
        "liveness_model": MODEL_NAME,
        "scrfd": face_mod.DETECTOR_NAME,
        "phase_a_seconds": args.phase_a_seconds,
        "phase_b_seconds": args.phase_b_seconds,
        "sample_interval_s": args.sample_interval,
        "capture_info": capture_info,
        "phase_A_stationary": {
            "stats": stats_a,
            "samples": [asdict(r) for r in records_a],
        },
        "phase_B_head_motion": {
            "stats": stats_b,
            "samples": [asdict(r) for r in records_b],
        },
        "comparison": {
            "real_score_mean_delta_B_minus_A": (
                (stats_b["real_score_mean"] or 0) - (stats_a["real_score_mean"] or 0)
                if stats_a["real_score_mean"] is not None
                and stats_b["real_score_mean"] is not None
                else None
            ),
            "laplacian_var_mean_delta_B_minus_A": (
                (stats_b["laplacian_var_mean"] or 0) - (stats_a["laplacian_var_mean"] or 0)
                if stats_a["laplacian_var_mean"] is not None
                and stats_b["laplacian_var_mean"] is not None
                else None
            ),
            "yaw_proxy_stdev_A": stats_a.get("yaw_proxy_stdev"),
            "yaw_proxy_stdev_B": stats_b.get("yaw_proxy_stdev"),
        },
        "notes": [
            "Rendered diagnostic only; scores are raw anti-spoof-mn3 outputs.",
            "landmark_yaw_proxy is approximate (nose vs eye midline / inter-eye).",
        ],
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_lines = [
        "--- VIGIL head-motion liveness diagnostic ---",
        f"Run folder: {run_dir}",
        f"Device: {device_name}",
        "",
        "Phase A (stationary):",
        f"  scored: {stats_a['face_ok_scored']}/{stats_a['samples_total']}",
        f"  mean REAL: {stats_a['real_score_mean']}",
        f"  provisional: {stats_a['provisional_outcome_counts']}",
        "",
        "Phase B (head motion):",
        f"  scored: {stats_b['face_ok_scored']}/{stats_b['samples_total']}",
        f"  mean REAL: {stats_b['real_score_mean']}",
        f"  provisional: {stats_b['provisional_outcome_counts']}",
        "",
        f"Contact sheet: {contact_path}",
        f"Full report: {report_path}",
    ]
    summary_path = run_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()

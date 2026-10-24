"""Profile FriendAuthPipeline stage latency (no camera required).

Uses local enrollment photos for Seif recognition and a blank/textured frame
for NO_FACE early-exit. Writes a JSON report under data/ (gitignored).

Does not upload biometric data. Does not change models or thresholds.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from face_bridge import load_face_detector_module
from facenet_identity import DEFAULT_ENROLLMENT_PATH, load_enrollments
from live_runtime import configure_tf_threads_for_inference


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _summary(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "max": float("nan"), "mean": float("nan")}
    s = sorted(vals)
    return {
        "n": float(len(s)),
        "median": float(statistics.median(s)),
        "p95": float(_pct(s, 95)),
        "max": float(max(s)),
        "mean": float(statistics.mean(s)),
        "min": float(min(s)),
    }


def _load_seif_frames(enrollment_path: Path, limit: int) -> list[tuple[str, np.ndarray]]:
    db = load_enrollments(enrollment_path)
    rec = (db.get("identities") or {}).get("Seif") or {}
    paths = list(rec.get("source_images") or [])
    out: list[tuple[str, np.ndarray]] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        out.append((path.name, img))
        if len(out) >= limit:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile VIGIL auth pipeline latency")
    parser.add_argument("--enrollment-db", type=Path, default=DEFAULT_ENROLLMENT_PATH)
    parser.add_argument("--images", type=int, default=8, help="Seif photos to score")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per image after warm")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "auth_latency_profile.json",
    )
    args = parser.parse_args()

    tf_status = configure_tf_threads_for_inference()
    print(f"[INFO] {tf_status}")

    from auth_pipeline import FriendAuthPipeline  # after TF thread config

    face_mod = load_face_detector_module()
    detector = face_mod.FaceDetector()
    pipeline = FriendAuthPipeline(detector, enrollment_path=args.enrollment_db)

    t_cold0 = time.perf_counter()
    init = pipeline.warm_up()
    cold_ms = (time.perf_counter() - t_cold0) * 1000.0
    print(
        f"[INFO] cold warm_up={cold_ms:.0f} ms "
        f"fasnet_backend={getattr(pipeline.fasnet, 'backend', '?')} "
        f"facenet_backend={getattr(pipeline.facenet, 'backend', '?')} "
        f"fasnet_init={init.get('fasnet_init_ms')} facenet_init={init.get('facenet_init_ms')}"
    )

    frames = _load_seif_frames(args.enrollment_db, args.images)
    if not frames:
        print("[ERROR] No Seif enrollment photos found for profiling", flush=True)
        return 1

    # Textured empty scene for NO_FACE early exit
    rng = np.random.default_rng(0)
    no_face = rng.integers(40, 200, (480, 640, 3), dtype=np.uint8)

    buckets: dict[str, list[dict[str, Any]]] = {
        "complete_identity_eval": [],
        "early_no_face": [],
        "spoof_or_deny": [],
        "error": [],
    }

    # Cold first pass on first image (already warmed models; first eval may still be slower)
    name0, img0 = frames[0]
    t0 = time.perf_counter()
    first = pipeline.process_verified_frame(img0)
    first_wall = (time.perf_counter() - t0) * 1000.0
    print(
        f"[INFO] first_eval wall={first_wall:.1f} ms total={first.total_ms:.1f} "
        f"scrfd={first.scrfd_ms:.1f} fas={first.fasnet_ms:.1f} fn={first.facenet_ms:.1f} "
        f"decision={first.outcome.decision.value} reason={first.outcome.reason} "
        f"id={first.outcome.identity} err={getattr(first.identity, 'error', None)}"
    )

    def record(tag: str, result, wall_ms: float, label: str) -> None:
        entry = {
            "label": label,
            "wall_ms": wall_ms,
            "total_ms": result.total_ms,
            "scrfd_ms": result.scrfd_ms,
            "fasnet_ms": result.fasnet_ms,
            "facenet_ms": result.facenet_ms,
            "fasnet_v2_ms": getattr(result, "fasnet_v2_ms", None),
            "fasnet_v1se_ms": getattr(result, "fasnet_v1se_ms", None),
            "match_ms": getattr(result, "match_ms", None),
            "decision": result.outcome.decision.value,
            "reason": result.outcome.reason,
            "recognized": result.outcome.recognized,
            "identity": result.outcome.identity,
            "identity_error": getattr(result.identity, "error", None) if result.identity else None,
            "best_distance": (
                result.identity.best_distance if result.identity is not None else None
            ),
            "liveness": result.outcome.liveness_class,
            "fasnet_score": (
                result.fasnet.antispoof_score if result.fasnet is not None else None
            ),
        }
        if result.identity is not None and result.identity.error:
            buckets["error"].append(entry)
        elif result.outcome.reason == "no_face":
            buckets["early_no_face"].append(entry)
        elif result.outcome.reason == "spoof":
            buckets["spoof_or_deny"].append(entry)
        elif result.fasnet is not None and result.identity is not None and not result.identity.error:
            buckets["complete_identity_eval"].append(entry)
        else:
            buckets["spoof_or_deny"].append(entry)
        print(
            f"  {tag} {label}: wall={wall_ms:.1f} total={result.total_ms:.1f} "
            f"scrfd={result.scrfd_ms:.1f} fas={result.fasnet_ms:.1f} "
            f"fn={result.facenet_ms:.1f} {result.outcome.decision.value}/{result.outcome.reason}"
            + (
                f" dist={result.identity.best_distance:.2f}"
                if result.identity and result.identity.best_distance is not None
                else ""
            )
        )

    print("[INFO] Steady-state Seif photos...")
    for name, img in frames:
        for r in range(args.repeats):
            t0 = time.perf_counter()
            result = pipeline.process_verified_frame(img)
            wall = (time.perf_counter() - t0) * 1000.0
            record(f"seif#{r+1}", result, wall, name)

    print("[INFO] NO_FACE early-exit frames...")
    for r in range(max(5, args.repeats)):
        t0 = time.perf_counter()
        result = pipeline.process_verified_frame(no_face)
        wall = (time.perf_counter() - t0) * 1000.0
        record(f"noface#{r+1}", result, wall, "textured_empty")

    def stage_stats(entries: list[dict[str, Any]], key: str) -> dict[str, float]:
        return _summary([float(e[key]) for e in entries if e.get(key) is not None])

    report: dict[str, Any] = {
        "hardware_note": "AMD Ryzen 5 5600H / RTX 3050 Ti / 16GB — TF GPU unavailable on native Windows TF>=2.11",
        "tf_threads": tf_status,
        "cold_warmup_ms": cold_ms,
        "first_eval_wall_ms": first_wall,
        "first_eval": {
            "total_ms": first.total_ms,
            "scrfd_ms": first.scrfd_ms,
            "fasnet_ms": first.fasnet_ms,
            "facenet_ms": first.facenet_ms,
            "decision": first.outcome.decision.value,
            "reason": first.outcome.reason,
            "identity": first.outcome.identity,
            "identity_error": getattr(first.identity, "error", None) if first.identity else None,
        },
        "buckets": {},
    }

    for name, entries in buckets.items():
        report["buckets"][name] = {
            "count": len(entries),
            "wall_ms": stage_stats(entries, "wall_ms"),
            "total_ms": stage_stats(entries, "total_ms"),
            "scrfd_ms": stage_stats(entries, "scrfd_ms"),
            "fasnet_ms": stage_stats(entries, "fasnet_ms"),
            "facenet_ms": stage_stats(entries, "facenet_ms"),
            "fasnet_v2_ms": stage_stats(entries, "fasnet_v2_ms"),
            "fasnet_v1se_ms": stage_stats(entries, "fasnet_v1se_ms"),
            "match_ms": stage_stats(entries, "match_ms"),
            "samples": entries,
        }

    complete = buckets["complete_identity_eval"]
    if complete:
        med = report["buckets"]["complete_identity_eval"]["total_ms"]["median"]
        report["auth_rate_hz_from_median_total"] = (
            1000.0 / med if med and med > 0 else float("nan")
        )
        recognized = sum(1 for e in complete if e.get("recognized"))
        report["complete_recognized_fraction"] = recognized / len(complete)
        errors = sum(1 for e in complete if e.get("identity_error"))
        report["complete_identity_error_count"] = errors

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {args.out}")

    # Console summary table
    print("\n=== Baseline summary (complete identity evaluations) ===")
    c = report["buckets"]["complete_identity_eval"]
    for key in ("scrfd_ms", "fasnet_ms", "facenet_ms", "total_ms", "wall_ms"):
        s = c[key]
        print(
            f"  {key:12s} n={int(s['n']):3d}  median={s['median']:7.1f}  "
            f"p95={s['p95']:7.1f}  max={s['max']:7.1f} ms"
        )
    if "auth_rate_hz_from_median_total" in report:
        print(f"  auth_rate~={report['auth_rate_hz_from_median_total']:.2f} Hz (from median total)")
    print(f"  recognized_fraction={report.get('complete_recognized_fraction')}")
    print(f"  identity_errors_in_complete={report.get('complete_identity_error_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Same-weight FaceNet ONNX vs TF: forward timing, embedding + decision parity.

Uses Seif enrollment photos locally. Writes data/facenet_onnx_bench.json (gitignored).
Does not modify enrollments.json or thresholds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from facenet_backends import (
    DEFAULT_ONNX_PATH,
    OnnxFaceNetForward,
    TfFaceNetForward,
    preprocess_facenet_input,
)
from facenet_identity import (
    DEFAULT_ENROLLMENT_PATH,
    MATCH_THRESHOLD,
    FaceNetIdentity,
    build_enrollment_index,
    load_enrollments,
    match_identity,
)
from live_runtime import configure_tf_threads_for_inference


def _summary(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": float(len(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "min": float(np.min(a)),
    }


def _load_frames(limit: int) -> list[tuple[str, np.ndarray]]:
    db = load_enrollments(DEFAULT_ENROLLMENT_PATH)
    paths = list((db.get("identities") or {}).get("Seif", {}).get("source_images") or [])
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
    out_path = Path(__file__).resolve().parent / "data" / "facenet_onnx_bench.json"
    print(configure_tf_threads_for_inference(), flush=True)
    if not DEFAULT_ONNX_PATH.is_file():
        print(f"[ERROR] missing ONNX at {DEFAULT_ONNX_PATH}", flush=True)
        return 2

    frames = _load_frames(16)
    if len(frames) < 4:
        print("[ERROR] need Seif source images", flush=True)
        return 2

    db = load_enrollments()
    index = build_enrollment_index(db)
    n_emb = len(((db.get("identities") or {}).get("Seif") or {}).get("embeddings") or [])
    print(f"[INFO] Seif enrollments={n_emb} frames={len(frames)} onnx={DEFAULT_ONNX_PATH}", flush=True)

    tf_fwd = TfFaceNetForward()
    onnx_fwd = OnnxFaceNetForward()
    tf_fwd.warm_up()
    onnx_fwd.warm_up()
    print(
        f"[INFO] warm tf_init={tf_fwd.init_ms:.0f} ms onnx_init={onnx_fwd.init_ms:.0f} ms",
        flush=True,
    )

    # Same-tensor forward bench (no detect/preprocess variance)
    sample = Path(__file__).resolve().parent / "data" / "facenet_sample_input.npy"
    if sample.is_file():
        batch0 = np.load(sample).astype(np.float32)
        if batch0.ndim == 3:
            batch0 = np.expand_dims(batch0, 0)
    else:
        batch0, _ = preprocess_facenet_input(frames[0][1])
        np.save(sample, batch0)

    for _ in range(3):
        tf_fwd.forward(batch0)
        onnx_fwd.forward(batch0)
    tf_fwd_ms: list[float] = []
    onnx_fwd_ms: list[float] = []
    for _ in range(20):
        _, ms = tf_fwd.forward(batch0)
        tf_fwd_ms.append(ms)
        _, ms = onnx_fwd.forward(batch0)
        onnx_fwd_ms.append(ms)
    e_tf, _ = tf_fwd.forward(batch0)
    e_onnx, _ = onnx_fwd.forward(batch0)
    same_tensor_max_abs = float(
        np.max(np.abs(np.asarray(e_tf, dtype=np.float64) - np.asarray(e_onnx, dtype=np.float64)))
    )
    same_tensor_l2 = float(
        np.linalg.norm(np.asarray(e_tf, dtype=np.float64) - np.asarray(e_onnx, dtype=np.float64))
    )
    print(
        f"[INFO] same-tensor forward TF med={np.median(tf_fwd_ms):.1f} "
        f"ONNX med={np.median(onnx_fwd_ms):.1f} max_abs={same_tensor_max_abs:.8f} "
        f"l2={same_tensor_l2:.8f}",
        flush=True,
    )

    # Full preprocess+forward + decision parity on SCRFD-free crops (enrollment photos)
    emb_l2: list[float] = []
    emb_max_abs: list[float] = []
    dist_delta: list[float] = []
    decision_mismatch = 0
    near_threshold: list[dict[str, Any]] = []
    tf_total: list[float] = []
    onnx_total: list[float] = []
    onnx_forward_only: list[float] = []
    pre_ms: list[float] = []

    # Warm full path once
    for name, img in frames[:2]:
        batch, _ = preprocess_facenet_input(img)
        tf_fwd.forward(batch)
        onnx_fwd.forward(batch)

    for name, img in frames:
        batch, pre = preprocess_facenet_input(img)
        pre_ms.append(pre.extract_faces_ms + pre.resize_pad_ms + pre.rgb_to_bgr_ms + pre.normalize_ms)

        et, t_ms = tf_fwd.forward(batch)
        eo, o_ms = onnx_fwd.forward(batch)
        onnx_forward_only.append(o_ms)
        tf_total.append(pre_ms[-1] + t_ms)
        onnx_total.append(pre_ms[-1] + o_ms)

        a = np.asarray(et, dtype=np.float64)
        b = np.asarray(eo, dtype=np.float64)
        l2 = float(np.linalg.norm(a - b))
        mx = float(np.max(np.abs(a - b)))
        emb_l2.append(l2)
        emb_max_abs.append(mx)

        mt = match_identity(et, db, threshold=MATCH_THRESHOLD, enrollment_index=index)
        mo = match_identity(eo, db, threshold=MATCH_THRESHOLD, enrollment_index=index)
        dt = mt.best_distance if mt.best_distance is not None else float("nan")
        do = mo.best_distance if mo.best_distance is not None else float("nan")
        if mt.recognized != mo.recognized or mt.identity != mo.identity:
            decision_mismatch += 1
        if np.isfinite(dt) and np.isfinite(do):
            dist_delta.append(abs(dt - do))
            for label, dist, rec in (("tf", dt, mt.recognized), ("onnx", do, mo.recognized)):
                if abs(dist - MATCH_THRESHOLD) < 1.5:
                    near_threshold.append(
                        {
                            "image": name,
                            "backend": label,
                            "distance": dist,
                            "recognized": rec,
                            "gap_to_9": dist - MATCH_THRESHOLD,
                        }
                    )
        print(
            f"  {name}: l2={l2:.6e} max_abs={mx:.6e} "
            f"dist_tf={dt:.4f} dist_onnx={do:.4f} "
            f"rec={mt.recognized}/{mo.recognized} "
            f"fwd_tf={t_ms:.0f} fwd_onnx={o_ms:.0f}",
            flush=True,
        )

    # Identity API end-to-end (selected backends)
    id_tf = FaceNetIdentity(backend="tf")
    id_onnx = FaceNetIdentity(backend="onnx")
    id_tf.warm_up()
    id_onnx.warm_up()
    api_tf_ms: list[float] = []
    api_onnx_ms: list[float] = []
    api_mismatch = 0
    for name, img in frames:
        et, tms, e1 = id_tf.embed_bgr(img)
        eo, oms, e2 = id_onnx.embed_bgr(img)
        assert e1 is None and e2 is None and et and eo
        api_tf_ms.append(tms)
        api_onnx_ms.append(oms)
        mt = match_identity(et, db, enrollment_index=index)
        mo = match_identity(eo, db, enrollment_index=index)
        if mt.recognized != mo.recognized or mt.identity != mo.identity:
            api_mismatch += 1

    report = {
        "onnx_path": str(DEFAULT_ONNX_PATH),
        "seif_embeddings": n_emb,
        "n_frames": len(frames),
        "same_tensor": {
            "tf_forward_ms": _summary(tf_fwd_ms),
            "onnx_forward_ms": _summary(onnx_fwd_ms),
            "max_abs": same_tensor_max_abs,
            "l2": same_tensor_l2,
        },
        "shared_preprocess_path": {
            "preprocess_ms": _summary(pre_ms),
            "tf_total_ms": _summary(tf_total),
            "onnx_forward_ms": _summary(onnx_forward_only),
            "onnx_total_ms": _summary(onnx_total),
            "emb_l2": _summary(emb_l2),
            "emb_max_abs": _summary(emb_max_abs),
            "distance_abs_delta": _summary(dist_delta),
            "decision_mismatches": decision_mismatch,
            "near_threshold_hits": near_threshold,
        },
        "FaceNetIdentity_api": {
            "tf_embed_ms": _summary(api_tf_ms),
            "onnx_embed_ms": _summary(api_onnx_ms),
            "decision_mismatches": api_mismatch,
        },
        "adopt_recommendation": (
            "onnx_default_ok"
            if (
                decision_mismatch == 0
                and api_mismatch == 0
                and (not emb_max_abs or max(emb_max_abs) < 1e-3)
                and (not dist_delta or max(dist_delta) < 1e-3)
            )
            else "keep_tf_default"
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["same_tensor"], indent=2), flush=True)
    print(json.dumps(report["shared_preprocess_path"], indent=2), flush=True)
    print(json.dumps(report["FaceNetIdentity_api"], indent=2), flush=True)
    print(f"[OK] adopt={report['adopt_recommendation']} → {out_path}", flush=True)
    return 0 if report["adopt_recommendation"] == "onnx_default_ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage-level FaceNet bottleneck diagnosis + direct-path parity (local only).

Reproduces DeepFace 0.0.102 represent() preprocessing for Facenet with
normalization=base (identity after resize_image /255).

Writes JSON under data/ (gitignored). Does not upload biometric data.
Does not change the default auth path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from facenet_identity import DEFAULT_ENROLLMENT_PATH, FaceNetIdentity, load_enrollments, match_identity
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


def preprocess_like_deepface_represent(
    image_bgr: np.ndarray,
    *,
    enforce_detection: bool = False,
    align: bool = True,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    """
    Mirror deepface.modules.representation.represent for a single BGR ndarray
    with detector_backend=opencv, normalization=base, model input 160x160.
    """
    from deepface.modules import detection, modeling, preprocessing

    times: dict[str, float] = {}
    t0 = time.perf_counter()
    model = modeling.build_model(task="facial_recognition", model_name="Facenet")
    times["build_model_cached_ms"] = (time.perf_counter() - t0) * 1000.0
    target_size = model.input_shape  # (160, 160)

    t0 = time.perf_counter()
    img_objs = detection.extract_faces(
        img_path=image_bgr,
        detector_backend="opencv",
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=0,
        anti_spoofing=False,
        max_faces=1,
    )
    times["extract_faces_ms"] = (time.perf_counter() - t0) * 1000.0
    if not img_objs:
        raise RuntimeError("extract_faces returned no faces")

    face_rgb = img_objs[0]["face"]
    region = img_objs[0]["facial_area"]
    conf = float(img_objs[0].get("confidence") or 0.0)

    t0 = time.perf_counter()
    face_bgr = face_rgb[:, :, ::-1]
    times["rgb_to_bgr_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    # resize_image expects BGR uint8-ish; returns (1,H,W,3) float32 in [0,1]
    batch = preprocessing.resize_image(
        img=face_bgr, target_size=(target_size[1], target_size[0])
    )
    times["resize_pad_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    batch = preprocessing.normalize_input(img=batch, normalization="base")
    times["normalize_ms"] = (time.perf_counter() - t0) * 1000.0

    meta = {
        "facial_area": region,
        "face_confidence": conf,
        "input_shape": list(batch.shape),
        "input_dtype": str(batch.dtype),
        "input_min": float(batch.min()),
        "input_max": float(batch.max()),
        "input_mean": float(batch.mean()),
    }
    return batch, times, meta


def forward_tf(model_client: Any, batch: np.ndarray) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    emb = model_client.forward(batch)
    ms = (time.perf_counter() - t0) * 1000.0
    if isinstance(emb[0], list):
        emb = emb[0]
    return list(emb), ms


def main() -> int:
    print(configure_tf_threads_for_inference(), flush=True)
    from deepface.modules import modeling

    db = load_enrollments(DEFAULT_ENROLLMENT_PATH)
    paths = list((db.get("identities") or {}).get("Seif", {}).get("source_images") or [])[:8]
    if len(paths) < 2:
        print("[ERROR] need Seif enrollment photos", flush=True)
        return 1

    # SCRFD crops like auth_pipeline (friend path)
    from face_bridge import load_face_detector_module

    det = load_face_detector_module().FaceDetector()
    crops: list[tuple[str, np.ndarray]] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        d = det.detect(img)
        if d.face_count != 1:
            continue
        f = d.faces[0]
        x1, y1, x2, y2 = map(int, (f.x1, f.y1, f.x2, f.y2))
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size:
            crops.append((Path(p).name, crop))

    print(f"[INFO] prepared {len(crops)} SCRFD crops", flush=True)
    model = modeling.build_model(task="facial_recognition", model_name="Facenet")
    legacy = FaceNetIdentity()
    legacy.warm_up()

    stage_acc: dict[str, list[float]] = {
        "extract_faces_ms": [],
        "rgb_to_bgr_ms": [],
        "resize_pad_ms": [],
        "normalize_ms": [],
        "tf_forward_ms": [],
        "direct_total_ms": [],
        "deepface_represent_ms": [],
        "match_ms": [],
        "emb_l2_vs_deepface": [],
        "dist_direct": [],
        "dist_deepface": [],
    }
    decision_same = 0
    samples_out: list[dict[str, Any]] = []
    first_input: np.ndarray | None = None

    for name, crop in crops:
        # Warm stages once per crop then measure
        for i in range(3):
            batch, st, meta = preprocess_like_deepface_represent(crop, enforce_detection=False)
            emb_d, fwd_ms = forward_tf(model, batch)
            t_df0 = time.perf_counter()
            emb_l, df_ms, err = legacy.embed_bgr(crop, enforce_detection=False)
            df_wall = (time.perf_counter() - t_df0) * 1000.0
            if err or emb_l is None:
                print(f"[WARN] deepface fail {name}: {err}", flush=True)
                continue
            if i == 0:
                continue  # warm
            t_m = time.perf_counter()
            m_d = match_identity(emb_d, db)
            match_ms = (time.perf_counter() - t_m) * 1000.0
            m_l = match_identity(emb_l, db)
            l2 = float(np.linalg.norm(np.asarray(emb_d) - np.asarray(emb_l)))
            for k, v in st.items():
                if k.endswith("_ms") and k in stage_acc:
                    stage_acc[k].append(v)
            stage_acc["tf_forward_ms"].append(fwd_ms)
            stage_acc["direct_total_ms"].append(
                st["extract_faces_ms"]
                + st["rgb_to_bgr_ms"]
                + st["resize_pad_ms"]
                + st["normalize_ms"]
                + fwd_ms
            )
            stage_acc["deepface_represent_ms"].append(df_wall)
            stage_acc["match_ms"].append(match_ms)
            stage_acc["emb_l2_vs_deepface"].append(l2)
            stage_acc["dist_direct"].append(float(m_d.best_distance or -1))
            stage_acc["dist_deepface"].append(float(m_l.best_distance or -1))
            same = (
                m_d.recognized == m_l.recognized
                and m_d.identity == m_l.identity
            )
            decision_same += int(same)
            if first_input is None:
                first_input = batch.copy()
            samples_out.append(
                {
                    "name": name,
                    "stages": {**st, "tf_forward_ms": fwd_ms, "deepface_ms": df_wall},
                    "meta": meta,
                    "emb_l2": l2,
                    "dist_direct": m_d.best_distance,
                    "dist_deepface": m_l.best_distance,
                    "recognized_same": same,
                    "identity_direct": m_d.identity,
                    "identity_deepface": m_l.identity,
                }
            )
            print(
                f"  {name}: extract={st['extract_faces_ms']:.1f} resize={st['resize_pad_ms']:.1f} "
                f"fwd={fwd_ms:.1f} direct_tot={stage_acc['direct_total_ms'][-1]:.1f} "
                f"deepface={df_wall:.1f} l2={l2:.6f} dist_d={m_d.best_distance:.3f} "
                f"dist_l={m_l.best_distance:.3f} same={same}",
                flush=True,
            )

    report = {
        "note": "SCRFD crop → opencv extract_faces+align → resize/pad/255 → TF Facenet forward",
        "stages": {k: _summary(v) for k, v in stage_acc.items()},
        "decision_same_count": decision_same,
        "sample_count": len(samples_out),
        "samples": samples_out,
    }
    out = Path(__file__).resolve().parent / "data" / "facenet_stage_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if first_input is not None:
        np.save(out.with_name("facenet_sample_input.npy"), first_input)
        print(f"[OK] saved input tensor {first_input.shape} → {out.with_name('facenet_sample_input.npy')}", flush=True)
    print(f"[OK] {out}", flush=True)
    print("=== Stage medians (ms) ===", flush=True)
    for k, s in report["stages"].items():
        if s.get("n"):
            print(f"  {k:28s} median={s['median']:8.2f}  p95={s['p95']:8.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

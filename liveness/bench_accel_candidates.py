"""Micro-benchmarks for acceleration candidates (parity + latency).

Compares:
  - TF FasNet (current wrapper)
  - ONNX FasNet ensemble (same crops; softmax baked into ONNX graphs)
  - FaceNet represent opencv vs skip (embedding L2 / decision impact)

Does not change the default pipeline. Writes JSON under data/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from face_bridge import load_face_detector_module
from facenet_identity import DEFAULT_ENROLLMENT_PATH, FaceNetIdentity, load_enrollments, match_identity
from fasnet_liveness import FasNetLiveness, bbox_xyxy_to_xywh, scores_from_ensemble_prediction
from live_runtime import configure_tf_threads_for_inference


def main() -> int:
    print(configure_tf_threads_for_inference(), flush=True)
    db = load_enrollments(DEFAULT_ENROLLMENT_PATH)
    src = Path((db["identities"]["Seif"]["source_images"])[0])
    img = cv2.imread(str(src))
    assert img is not None

    face_mod = load_face_detector_module()
    det = face_mod.FaceDetector().detect(img)
    assert det.face_count == 1
    face = det.faces[0]
    xywh = bbox_xyxy_to_xywh(face.bbox)
    x1, y1, x2, y2 = map(int, (face.x1, face.y1, face.x2, face.y2))
    crop = img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]

    # --- TF FasNet ---
    tf_fas = FasNetLiveness()
    tf_fas.warm_up()
    tf_times = []
    tf_scores = []
    for i in range(6):
        t0 = time.perf_counter()
        r = tf_fas.analyze_full_frame(img, face.bbox)
        dt = (time.perf_counter() - t0) * 1000.0
        if i > 0:
            tf_times.append(dt)
            tf_scores.append(
                {
                    "is_real": r.is_real,
                    "score": r.antispoof_score,
                    "avg": r.class_scores_avg,
                    "v2": r.mini_v2_ms,
                    "v1se": r.mini_v1se_ms,
                    "pre": r.preprocess_ms,
                }
            )
    print(f"TF FasNet median={np.median(tf_times):.1f} ms scores0={tf_scores[0]}", flush=True)

    # --- ONNX FasNet ---
    from deepface.commons.onnx_utils import build_session, run_session
    from deepface.models.spoofing.FasNetUtils import crop as fas_crop

    first = build_session(
        file_name="2.7_80x80_MiniFASNetV2.onnx",
        source_url=[
            "https://github.com/serengil/deepface_models/releases/download/v1.0/2.7_80x80_MiniFASNetV2.onnx",
            "https://huggingface.co/serengil/deepface/resolve/main/2.7_80x80_MiniFASNetV2.onnx",
        ],
    )
    second = build_session(
        file_name="4_0_0_80x80_MiniFASNetV1SE.onnx",
        source_url=[
            "https://github.com/serengil/deepface_models/releases/download/v1.0/4_0_0_80x80_MiniFASNetV1SE.onnx",
            "https://huggingface.co/serengil/deepface/resolve/main/4_0_0_80x80_MiniFASNetV1SE.onnx",
        ],
    )

    def onnx_row(frame, xywh_):
        x, y, w, h = xywh_
        a = fas_crop(frame, (x, y, w, h), 2.7, 80, 80)
        b = fas_crop(frame, (x, y, w, h), 4, 80, 80)
        # Softmax baked into ONNX graphs — do not apply again.
        r1 = np.asarray(run_session(first, a), dtype=np.float64).reshape(-1)
        r2 = np.asarray(run_session(second, b), dtype=np.float64).reshape(-1)
        return r1 + r2

    onnx_times = []
    onnx_scores = []
    for i in range(6):
        t0 = time.perf_counter()
        row = onnx_row(img, xywh)
        is_real, score, avg, real_avg, spoof_c = scores_from_ensemble_prediction(row)
        dt = (time.perf_counter() - t0) * 1000.0
        if i > 0:
            onnx_times.append(dt)
            onnx_scores.append({"is_real": is_real, "score": score, "avg": avg})
    print(f"ONNX FasNet median={np.median(onnx_times):.1f} ms scores0={onnx_scores[0]}", flush=True)

    # Compare TF vs ONNX decisions / score deltas
    tf0 = tf_scores[0]
    on0 = onnx_scores[0]
    avg_delta = [
        abs(a - b) for a, b in zip(tf0["avg"] or (0, 0, 0), on0["avg"] or (0, 0, 0))
    ]
    print(
        f"FasNet parity is_real_match={tf0['is_real']==on0['is_real']} "
        f"score_delta={abs((tf0['score'] or 0)-(on0['score'] or 0)):.6f} "
        f"max_avg_delta={max(avg_delta):.6f}",
        flush=True,
    )

    # --- FaceNet opencv vs skip ---
    from deepface import DeepFace

    fn = FaceNetIdentity()
    fn.warm_up()

    def timed_represent(backend: str, n: int = 4):
        times = []
        emb = None
        for i in range(n):
            t0 = time.perf_counter()
            out = DeepFace.represent(
                img_path=crop,
                model_name="Facenet",
                enforce_detection=False,
                detector_backend=backend,
            )
            dt = (time.perf_counter() - t0) * 1000.0
            emb = list(out[0]["embedding"])
            if i > 0:
                times.append(dt)
        return times, emb

    t_cv, emb_cv = timed_represent("opencv")
    t_sk, emb_sk = timed_represent("skip")
    l2 = float(np.linalg.norm(np.asarray(emb_cv) - np.asarray(emb_sk)))
    m_cv = match_identity(emb_cv, db)
    m_sk = match_identity(emb_sk, db)
    print(
        f"FaceNet opencv median={np.median(t_cv):.1f} skip median={np.median(t_sk):.1f} "
        f"emb_L2={l2:.4f} dist_cv={m_cv.best_distance:.3f} dist_sk={m_sk.best_distance:.3f} "
        f"recog_cv={m_cv.recognized} recog_sk={m_sk.recognized} "
        f"decision_same={m_cv.recognized==m_sk.recognized and m_cv.identity==m_sk.identity}",
        flush=True,
    )

    report = {
        "tf_fasnet_ms": {"median": float(np.median(tf_times)), "samples": tf_times, "detail0": tf_scores[0]},
        "onnx_fasnet_ms": {
            "median": float(np.median(onnx_times)),
            "samples": onnx_times,
            "detail0": {
                "is_real": bool(on0["is_real"]),
                "score": float(on0["score"]),
                "avg": [float(x) for x in on0["avg"]],
            },
        },
        "fasnet_parity": {
            "is_real_match": bool(tf0["is_real"] == on0["is_real"]),
            "score_delta": abs(float(tf0["score"] or 0) - float(on0["score"] or 0)),
            "max_class_avg_delta": float(max(avg_delta)),
        },
        "facenet": {
            "opencv_ms_median": float(np.median(t_cv)),
            "skip_ms_median": float(np.median(t_sk)),
            "embedding_l2": l2,
            "dist_opencv": m_cv.best_distance,
            "dist_skip": m_sk.best_distance,
            "recognized_same": bool(
                m_cv.recognized == m_sk.recognized and m_cv.identity == m_sk.identity
            ),
        },
    }
    out = Path(__file__).resolve().parent / "data" / "accel_candidate_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

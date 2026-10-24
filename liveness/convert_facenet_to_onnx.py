"""Export the *exact* DeepFace TF FaceNet-128 Keras weights to ONNX (experimental).

Isolates conversion: writes under data/facenet_onnx/ (gitignored).
Does not change the default auth backend.

Usage (from liveness/ with .venv_auth):
  python convert_facenet_to_onnx.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "data" / "facenet_onnx"
ONNX_NAME = "facenet128_deepface_vigil.onnx"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Prefer tf2onnx; fail clearly if missing.
    try:
        import tf2onnx
        import tensorflow as tf
    except ImportError as exc:
        print(
            f"[ERROR] Need tf2onnx + tensorflow in this env: {exc}\n"
            "Install into a disposable venv if unsure:\n"
            "  python -m venv .venv_facenet_onnx\n"
            "  .venv_facenet_onnx\\Scripts\\pip install tf2onnx tensorflow tf-keras onnx\n"
            "Then re-run this script with that interpreter after copying deepface weights.",
            flush=True,
        )
        return 2

    from deepface.modules import modeling
    from live_runtime import configure_tf_threads_for_inference

    print(configure_tf_threads_for_inference(), flush=True)
    client = modeling.build_model(task="facial_recognition", model_name="Facenet")
    keras_model = client.model
    print(
        f"[INFO] loaded {client.model_name} input={client.input_shape} "
        f"output={client.output_shape}",
        flush=True,
    )

    # Sanity: one forward
    dummy = np.zeros((1, 160, 160, 3), dtype=np.float32)
    ref = keras_model(dummy, training=False).numpy()
    print(f"[INFO] TF dummy forward shape={ref.shape} mean={ref.mean():.6f}", flush=True)

    spec = (tf.TensorSpec((None, 160, 160, 3), tf.float32, name="input"),)
    onnx_path = OUT_DIR / ONNX_NAME
    t_conv = time.perf_counter()
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=spec,
        opset=13,
        output_path=str(onnx_path),
    )
    conv_ms = (time.perf_counter() - t_conv) * 1000.0
    print(f"[OK] wrote {onnx_path} ({onnx_path.stat().st_size} bytes) in {conv_ms:.0f} ms", flush=True)

    # ORT parity on dummy
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    ort_out = sess.run([out_name], {inp_name: dummy})[0]
    max_abs = float(np.max(np.abs(ref - ort_out)))
    print(f"[INFO] dummy TF vs ORT max_abs={max_abs:.8f}", flush=True)

    meta = {
        "source": "deepface Facenet FaceNet128dClient Keras weights",
        "onnx_path": str(onnx_path),
        "opset": 13,
        "input_shape": [None, 160, 160, 3],
        "output_dim": 128,
        "dummy_max_abs_tf_vs_ort": max_abs,
        "convert_ms": conv_ms,
        "wall_ms": (time.perf_counter() - t0) * 1000.0,
        "tf_version": tf.__version__,
        "tf2onnx_version": getattr(tf2onnx, "__version__", "?"),
        "ort_version": ort.__version__,
        "input_name": inp_name,
        "output_name": out_name,
    }
    (OUT_DIR / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] meta → {OUT_DIR / 'export_meta.json'}", flush=True)
    if max_abs > 1e-3:
        print("[WARN] dummy parity looser than 1e-3 — inspect before adopting", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

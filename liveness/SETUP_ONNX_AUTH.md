# Reproduce optimized ONNX authentication (Windows / laptop)

Operational venv: `liveness/.venv_auth` (not `.venv_fsfm`).

## 1. Python environment

```powershell
cd liveness
python -m venv .venv_auth
.\.venv_auth\Scripts\pip install -r requirements-friend-auth.txt
```

## 2. SCRFD (required)

Runtime file (only this ONNX is used):

`face_detection/models/buffalo_m/buffalo_m/det_2.5g.onnx`

Restore if missing:

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL"
.\liveness\.venv_auth\Scripts\python -c "from insightface.utils.storage import download; download('models','buffalo_m', root=r'face_detection', force=True)"
```

SHA256 of `det_2.5g.onnx`: see `face_detection/models/MODEL_INFO.txt`.

## 3. FasNet ONNX (default liveness)

DeepFace downloads into `%USERPROFILE%\.deepface\weights\`:

- `2.7_80x80_MiniFASNetV2.onnx`
- `4_0_0_80x80_MiniFASNetV1SE.onnx`

(Also H5 copies for TF fallback.) First run of `live_friend_auth.py` or FasNet warm-up triggers download. Force:

```powershell
$env:VIGIL_FASNET_BACKEND='onnx'
```

TF fallback: `$env:VIGIL_FASNET_BACKEND='tf'`

## 4. FaceNet ONNX (default identity when export exists)

**Path (gitignored, ~87 MB):**  
`liveness/data/facenet_onnx/facenet128_deepface_vigil.onnx`

This is the **same** DeepFace FaceNet-128 Keras weights converted locally — not a third-party FaceNet.

### Export (once per machine / after wiping the file)

Requires DeepFace FaceNet H5 at `%USERPROFILE%\.deepface\weights\facenet_weights.h5` (auto-downloaded on first TF FaceNet use).

```powershell
cd liveness
$env:PYTHONIOENCODING='utf-8'
$env:TF_CPP_MIN_LOG_LEVEL='2'
.\.venv_auth\Scripts\python convert_facenet_to_onnx.py
```

Expect: `[OK] wrote ...\facenet128_deepface_vigil.onnx` and dummy TF vs ORT `max_abs` ≪ 1e-3.

### Runtime

```powershell
$env:VIGIL_FACENET_BACKEND='onnx'   # default when ONNX file + onnxruntime exist
$env:VIGIL_FACENET_BACKEND='tf'     # TensorFlow fallback (slower)
```

If the ONNX file is **missing**, the code silently falls back to TF. After setup, verify:

```powershell
.\.venv_auth\Scripts\python -c "from facenet_backends import resolve_facenet_backend; from fasnet_liveness import resolve_fasnet_backend; print(resolve_facenet_backend(), resolve_fasnet_backend())"
# expect: onnx onnx
```

## 5. Enrollment (local, never commit)

`liveness/data/enrollments.json` — Euclidean threshold **9.0** (UNVALIDATED).

## 6. Run auth

```powershell
.\.venv_auth\Scripts\python live_friend_auth.py --camera 0
```

HMAC path: `hmac/verifier_server.py --enable-friend-auth` + `trusted_capture.py`.

## 7. Calibration / Challenge 1

See `CALIBRATION.md` and `CHALLENGE1_PHYSICAL_TESTS.md`. Default FasNet policy remains **shadow** (0.85 gates not activated).

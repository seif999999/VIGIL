# VIGIL face detection baseline — SCRFD-2.5G_KPS

## Purpose

Locate faces in **authenticated** camera frames after HMAC verification. This layer provides:

- bounding box
- detection confidence
- five facial landmarks (for future EdgeFace alignment)

It does **not** perform liveness, identity matching, or HMAC.

## Selected model (frozen)

**SCRFD-2.5G_KPS** — InsightFace SCRFD 2.5G with five keypoints.

Runtime ONNX: `models/buffalo_m/buffalo_m/det_2.5g.onnx` from the official InsightFace **model-zoo** release package `buffalo_m.zip`. See `models/MODEL_INFO.txt` for source URL and SHA256.

Official WIDER FACE mAP numbers (Easy/Medium/Hard) are **literature values**, not re-measured here.

## Baseline face-count policy

| Detected faces | Decision |
|----------------|----------|
| 0 | `NO_FACE` |
| 1 | `FACE_OK` |
| 2+ | `MULTIPLE_FACES` |

No “largest face” selection — authentication requires exactly one face.

## Dependencies

- `opencv-python` (existing)
- `numpy`
- `onnxruntime` (CPU)
- `insightface` (SCRFD wrapper only; detection weights from model-zoo)

## Commands

**Download model (once):**

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL"
python -c "from insightface.utils.storage import download; download('models','buffalo_m', root=r'face_detection', force=False)"
```

**Static test:**

```powershell
cd face_detection
python test_detector.py --image ..\hmac\test_images\original_1.jpg
```

**Live detector only (no HMAC):**

```powershell
python live_detector.py --camera 0 --duration 25
```

**Benchmark:**

```powershell
python benchmark_detector.py --image ..\hmac\test_images\original_1.jpg --iterations 500
```

**HMAC-integrated verifier:**

```powershell
cd ..\hmac
python verifier_server.py --enable-face-detection
python trusted_capture.py --duration 25
```

Order: **TCP → FrameVerifier ACCEPT → decode JPEG → SCRFD**. Rejected HMAC frames never reach SCRFD.

**Security gate check (synthetic):**

```powershell
cd face_detection
python security_gate_test.py
```

**Evaluation set (when you add images + manifest.csv):**

```powershell
python evaluate_dataset.py --manifest evaluation\manifest.csv
```

## License note

InsightFace **code** is MIT. Official **pretrained models** are intended for non-commercial research. This thesis use is research/non-commercial.

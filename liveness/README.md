# VIGIL liveness

## Experimental friend-auth (FasNet + FaceNet)

On branch `friend-pipeline`, optional acceptance path after HMAC + SCRFD:

1. **FasNet** (`fasnet_liveness.py`) — DeepFace 0.0.102 ensemble (MiniFASNetV2@2.7 + MiniFASNetV1SE@4.0) on the **full frame** with SCRFD bbox as `(x,y,w,h)`.
2. **FaceNet-128** (`facenet_identity.py`) — local JSON enrollments; euclidean threshold **9.0** (`UNVALIDATED_FRIEND_THRESHOLD`).
3. **ACCEPT** only if REAL **and** recognized (`auth_decision.py` / `auth_pipeline.py`).

See root `README.md` for enroll / verifier commands. Legacy **anti-spoof-mn3** below remains for comparison (`--enable-liveness` / `--enable-visual-baseline`).

## Purpose (mn3 baseline)

Face **presentation attack detection** (liveness evidence) on a **single SCRFD face crop**, after HMAC verification and face-count policy.

This does **not** replace HMAC or prove digital frame integrity alone. It adds evidence that the face may be physically live vs print/screen spoof.

## Model

**anti-spoof-mn3** — MobileNetV3 binary classifier, trained on **CelebA-Spoof** (OpenVINO Open Model Zoo).

| Output index | Class |
|--------------|--------|
| 0 | REAL |
| 1 | SPOOF |

**Raw scores:** `real_score`, `spoof_score`, `predicted_class` = argmax. These are **not** calibrated VIGIL PASS/FAIL thresholds.

## Pipeline order (mandatory)

```
TCP → FrameVerifier ACCEPT → JPEG decode → SCRFD → face count
  0 faces → NO_FACE (no liveness)
  2+ faces → MULTIPLE_FACES (no liveness)
  1 face → crop bbox → anti-spoof-mn3
```

Rejected HMAC: **no decode, no SCRFD, no liveness.**

## Preprocessing (this baseline)

Official **ONNX** from OMZ (`anti-spoof-mn3.onnx`):

- Resize face crop to **128×128**
- **BGR → RGB**
- `(pixel - mean) / scale` per channel (OMZ Original Model — not ImageNet)
- NCHW float32 → `actual_input_1`

Face crop: **direct SCRFD bbox** (clamped), no expansion in this baseline.

## Limitations (document for thesis)

- HMAC + SCRFD proves **integrity after signing** and **one face location**, not physical liveness.
- anti-spoof-mn3 is trained on CelebA-Spoof; scores on your webcam may differ.
- Static photos can score as SPOOF even for a real person — context matters.
- You must run **REAL / PRINT / SCREEN** manual tests locally; raw scores only.

## Download model

```powershell
Invoke-WebRequest -Uri "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/anti-spoof-mn3/anti-spoof-mn3.onnx" -OutFile "liveness\models\anti-spoof-mn3.onnx"
```

See `models/MODEL_INFO.txt` for checksums.

## Dependencies

- `onnxruntime` (CPU)
- `opencv-python`, `numpy`
- SCRFD via existing `face_detection` (insightface)

## Commands

**Static:**

```powershell
cd liveness
python test_liveness.py --image ..\face_detection\my_face.jpeg
```

**Live (three optional sources — use one at a time):**

1. OpenCV camera index (laptop webcam):

```powershell
python live_liveness.py --camera 0 --duration 25
```

2. FFmpeg DirectShow named device (Windows virtual camera):

```powershell
python live_liveness.py --dshow-device "Nazmy (Windows Virtual Camera)" --width 1280 --height 720 --duration 25
```

3. Network stream URL (phone app must expose HTTP/MJPEG or RTSP reachable from the laptop):

```powershell
python live_liveness.py --stream-url "http://192.168.1.10:8080/video" --duration 25
python network_capture.py --stream-url "http://192.168.1.10:8080/video" --frames 10
```

4. Native Windows MediaCapture (connected camera / virtual camera — recommended for `Nazmy (Windows Virtual Camera)`):

```powershell
python windows_camera_capture.py --device-name "Nazmy (Windows Virtual Camera)" --frames 30
python live_liveness.py --windows-device "Nazmy (Windows Virtual Camera)" --duration 25
```

DirectShow and OpenCV often fail to read frames from this Windows virtual camera; **Windows MediaCapture** is the native path.

Use only one of: `--windows-device`, `--stream-url`, `--dshow-device` (plus default `--camera` when none of those are set).

**Manual scenarios (logging, no frame storage):**

```powershell
python live_liveness.py --duration 20 --scenario REAL --log-csv manual_logs\runs.csv
python live_liveness.py --duration 20 --scenario PRINT --log-csv manual_logs\runs.csv
python live_liveness.py --duration 20 --scenario SCREEN --log-csv manual_logs\runs.csv
```

**Benchmark:**

```powershell
python benchmark_liveness.py --image ..\face_detection\my_face.jpeg --iterations 500
```

**HMAC + SCRFD + liveness:**

```powershell
cd ..\hmac
python verifier_server.py --enable-face-detection --enable-liveness
python trusted_capture.py --duration 25
```

**Tests:**

```powershell
cd ..\liveness
python security_gate_test.py
python face_count_gate_test.py
```

## License note

Upstream model license MIT per OMZ; use consistent with non-commercial research thesis context.

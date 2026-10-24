# FaceNet identity calibration (physical)

Standalone camera tool to collect **GENUINE** vs **IMPOSTOR** FaceNet L2 distances
against the existing enrollment gallery. Uses the same SCRFD entry point as live auth
(`FaceDetector()` via `face_bridge.load_face_detector_module`). FasNet is not used.
Live auth, Challenge 1/2, enrollment, models, and `MATCH_THRESHOLD` (9.0) are unchanged.

## What is stored

JSONL under `liveness/data/identity_calibration/`:

- timestamps, anonymous `subject_id`, `session_id`, label (`GENUINE` | `IMPOSTOR`)
- face detection status (`FACE_OK` / `NO_FACE` / `MULTIPLE_FACES`)
- FaceNet `best_distance` (min L2 to gallery) and preview decision vs threshold 9.0

**Not** stored: camera images, embeddings, raw frames.

Only **MARKED** intervals are written (press `M`).

## Launch (PowerShell)

From the `liveness` folder, with your usual friend-auth / ONNX env active:

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\liveness"
python live_identity_calibration.py --label GENUINE --subject-id genuine_A --camera 0
```

Impostor session (separate process / separate file):

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\liveness"
python live_identity_calibration.py --label IMPOSTOR --subject-id impostor_01 --camera 0
```

Optional: `--facenet-backend onnx` (recommended when ONNX is set up), `--enrollment-db`, `--out-dir`, `--min-interval-ms`.

## Conducting a physical test

1. **Start** with the correct `--label` and a unique anonymous `--subject-id` (no legal names required).
2. Wait until the preview shows a stable face and a live L2 distance (or NO_FACE / MULTIPLE_FACES).
3. Press **M** to enter **MARKED** — distances and face status now append to the session JSONL.
4. Hold natural poses for ~10–30 s (or longer). Press **M** again to **PAUSED** between subjects or pose blocks.
5. Press **M** to resume for the next marked block in the same session if needed.
6. Press **Q** or **ESC** (or close the window) to stop, flush, and close the file.
7. Run a **new** launch for the other cohort (GENUINE vs IMPOSTOR) so files stay separated by label in the filename/`session_id`.

## Preview meaning

- Green box / RECOGNIZED: `best_distance < 9.0` (existing unvalidated friend threshold).
- Red box / UNKNOWN: distance ≥ 9.0 or FaceNet error.
- FaceNet runs on every usable single-face (`FACE_OK`) evaluation, independent of FasNet.

## After collection

Do not activate a new threshold from these files until offline analysis is reviewed.
Keep `thresholds_activated: false` in session meta as shipped.

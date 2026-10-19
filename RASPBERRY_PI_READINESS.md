# VIGIL — Raspberry Pi readiness (assessment only)

**Scope:** Integrated Windows pipeline on branch `updated-implementation`  
**Status:** Report only — **no Pi-specific code was implemented** (Pi model / OS unspecified).  
**Windows behavior, models, thresholds, and security logic are unchanged.**

---

## Pipeline under review

```
Trusted capture → HMAC → SCRFD (InsightFace ONNX) → DeepFace FasNet (TF) → FaceNet-128 (TF) → ACCEPT
Legacy parallel: anti-spoof-mn3 (ONNX Runtime)
```

---

## 1. Linux ARM64 component compatibility

| Component | Stack today | ARM64 Linux outlook |
|-----------|-------------|---------------------|
| **SCRFD** | InsightFace + `det_2.5g.onnx` via ONNX Runtime | **Feasible.** Prefer `onnxruntime` Linux aarch64 wheels; InsightFace generally works on Linux if ORT + OpenCV build. Avoid Windows-only packages. |
| **FasNet** | DeepFace 0.0.102 → TensorFlow MiniFASNet H5 | **Feasible but heavy.** Needs TF (or TF Lite conversion later). Official TF aarch64 support varies by Pi OS / Python version — plan for version pinning after target OS is known. |
| **FaceNet** | DeepFace `Facenet` TF weights (~92 MB) | Same as FasNet — **feasible, CPU-heavy**. |
| **anti-spoof-mn3** | ONNX Runtime CPU | **Strongest portable path** on Pi CPU. |
| **HMAC / protocol** | Pure Python (`hmac/protocol.py`) | **Unchanged** — fully portable. |
| **OpenCV** | `opencv-python` | Use Linux aarch64 wheels or distro OpenCV; GUI (`imshow`) optional on headless Pi. |

**Research-only PyTorch paths** (FLIP/FSFM/IADG): not required for ACCEPT; Pi port should **omit** them unless needed for thesis demos.

---

## 2. Windows-specific capture (must be replaced on Pi)

| Current Windows piece | Role | Pi replacement direction |
|----------------------|------|---------------------------|
| `liveness/windows_camera_capture.py` (WinRT MediaCapture) | Reliable Camo / named device | **Do not use on Pi.** |
| OpenCV `CAP_DSHOW` in `trusted_capture.py` / `live_friend_auth.py` | Webcam index | Use **V4L2** (`/dev/video*`) via OpenCV `CAP_V4L2` or default backend. |
| FFmpeg DirectShow helpers | Named virtual cameras | Optional: `libcamera` / `rpicam` pipeline or USB UVC camera. |
| `windows_capture_bridge` (C# WinRT) | Native bridge | **Windows-only** — leave as-is; Pi uses Python V4L2 path. |

**Unchanged on Pi:** JPEG encode → `create_frame_package` → TCP → `FrameVerifier` (HMAC boundary stays identical).

---

## 3. TF / PyTorch / ONNX / DeepFace on ARM64

| Runtime | Feasibility on Pi (ARM64) | Notes |
|---------|---------------------------|--------|
| **ONNX Runtime** | **Best default** | Official Linux aarch64 builds common; used by SCRFD + mn3. |
| **TensorFlow / tf-keras** | **Possible, painful** | Large install; RAM pressure on Pi 4 2–4 GB; version matrix depends on Debian/bookworm vs custom OS. DeepFace 0.0.102 expects TF path for FasNet/Facenet. |
| **DeepFace 0.0.102** | Depends on TF + OpenCV | Keep API, but may need lighter backends later (ONNX FasNet / TF Lite) **only after** Pi target is fixed. |
| **PyTorch** | Optional | Not required for production ACCEPT path; skip for Pi image size/RAM. |

Do **not** change Windows TF/DeepFace setup until a Pi target is chosen.

---

## 4. CPU performance & memory risks

Observed on current Windows experimental path (order-of-magnitude, not Pi measurements):

- Steady auth often **~0.8–1.5+ s/frame** with SCRFD + FasNet + FaceNet on desktop CPU/GPU mix.
- FaceNet warm-up alone can be **multi-second**; FasNet+FaceNet weights + TF runtime → **hundreds of MB–1+ GB** RSS risk.

**On a Raspberry Pi (CPU-only):**

- Expect **slower** than Windows laptop (often multi-second per full ACCEPT path) unless models are reduced or quantized.
- **RAM:** Pi 4 2 GB is tight with TF+DeepFace+InsightFace; **4 GB+ recommended**; 8 GB preferred for headroom.
- Continuous latest-frame design already helps (no queue backlog) — keep that pattern on Pi.
- Thermal throttling under sustained load is likely.

**No fabricated Pi FPS claims** — measure after hardware/OS selection.

---

## 5. Possible later adjustments (not implemented)

Only after Pi model + OS are specified:

1. Capture adapter: V4L2 / `libcamera` behind the same BGR→HMAC interface.
2. Prefer **ONNX Runtime** everywhere practical (SCRFD + mn3 already); evaluate ONNX/TFLite FasNet + FaceNet **without changing decision rules/thresholds**.
3. Optional: lower preview resolution for capture only (not model preprocess contracts).
4. Headless verifier (no `cv2.imshow`) for SSH/service mode.
5. Avoid shipping research PyTorch weights to the Pi image.

**Do not** change MATCH_THRESHOLD 9.0, FasNet ensemble logic, or HMAC fail-closed rules for performance.

---

## 6. What can remain completely unchanged

- `hmac/protocol.py` signing, verification, session/counter semantics  
- Face-count policy (exactly one face)  
- Auth decision: REAL ∧ recognized → ACCEPT  
- FasNet native crop scales (2.7 / 4.0) and FaceNet euclidean match rule  
- Enrollment JSON + SHA256 provenance format under `liveness/data/` (gitignored)  
- anti-spoof-mn3 as legacy comparison baseline  
- TCP package framing between trusted capture and verifier  

---

## Recommended migration plan (when Pi is chosen)

1. **Specify** board (e.g. Pi 4/5), RAM, OS (e.g. Raspberry Pi OS 64-bit), Python version, camera (USB UVC vs CSI).  
2. **Spike install:** ORT aarch64 + OpenCV + InsightFace SCRFD smoke test.  
3. **Spike install:** DeepFace/TF FaceNet+FasNet smoke **or** decide on ONNX conversion spike.  
4. **Add** Linux capture backend (new module); keep WinRT path for Windows.  
5. **Benchmark** SCRFD / FasNet / FaceNet / total ms on-device; tune only non-security knobs (resolution, continuous stale window).  
6. **Validate** HMAC + face-count + spoof/unknown fail-closed on Pi with the same LIVE physical matrix.  
7. **Document** Pi deps in a separate `requirements-pi.txt` — do not break Windows `requirements-friend-auth.txt`.

---

## Enrollment status (unchanged by this assessment)

Prior folder import on this branch remains: **`tester` = 30 embeddings** (5 prior + 25 imported; 2 no_face rejects; 5 ambiguous legacy captures skipped). Re-import is fingerprint-safe.

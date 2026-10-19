# VIGIL

Trusted-camera facial authentication research prototype.

**Baseline branch:** `main` @ `eb7856bee44cb9ce8036b2e6644fe2ef70b63174`  
**Experimental branch:** `friend-pipeline` — DeepFace FasNet + FaceNet identity on top of VIGIL HMAC/SCRFD.

## Pipeline

```
Trusted capture → HMAC sign → HMAC verify → SCRFD (exactly one face)
  → [experimental] DeepFace FasNet liveness
  → [experimental] FaceNet-128 identity
  → ACCEPT only if REAL and recognized
```

Legacy comparison path (unchanged): SCRFD → **anti-spoof-mn3** → provisional demo policy (not identity auth).

## Modules

| Folder | Role |
|--------|------|
| `hmac/` | HMAC-SHA256 frame packages, trusted capture, TCP verifier |
| `face_detection/` | SCRFD-2.5G_KPS face count gate |
| `liveness/` | mn3 baseline + FasNet/FaceNet experimental auth |
| `webots/` | Simulation capture bridge |

## Setup

1. Generate HMAC key (never commit):
   ```text
   cd hmac
   python keygen.py
   ```
2. Download models using `face_detection/models/MODEL_INFO.txt` and `liveness/models/MODEL_INFO.txt` (weights are gitignored).
3. Friend-auth Python deps:
   ```text
   cd liveness
   python -m venv .venv_auth
   .venv_auth\Scripts\pip install -r requirements-friend-auth.txt
   ```
   DeepFace pulls FasNet / FaceNet weights into `~/.deepface/weights/` on first use.

## Enroll (local JSON, gitignored)

```text
cd liveness
.venv_auth\Scripts\python enroll_identity.py --identity <name> --camera 0 --count 5
```

Enrollment file: `liveness/data/enrollments.json` (euclidean threshold **9.0**, status `UNVALIDATED_FRIEND_THRESHOLD`).

## LIVE integrated auth (HMAC)

Terminal A — verifier:

```text
cd hmac
python verifier_server.py --enable-friend-auth
```

Terminal B — trusted capture:

```text
cd hmac
python trusted_capture.py
```

Legacy mn3 visual baseline:

```text
python verifier_server.py --enable-visual-baseline
```

Direct camera demo (no HMAC — profiling only):

```text
cd liveness
.venv_auth\Scripts\python live_friend_auth.py --camera 0
```

## Tests

```text
cd hmac
python tests.py
python integrated_visual_baseline_test.py

cd ..\liveness
python -m unittest test_decision_policy.py test_auth_decision.py test_friend_auth_gates.py
python face_count_gate_test.py
python security_gate_test.py
```

FasNet parity (same bbox vs DeepFace `extract_faces`):

```text
.venv_auth\Scripts\python parity_fasnet_check.py --image <bgr_frame.png>
```

## Security notes

- Perception runs **only** after HMAC `ACCEPT`.
- Zero / multiple faces → DENY (no largest-face fallback).
- Spoof / unknown identity / inference errors → DENY or UNCERTAIN (fail closed for ACCEPT).
- Do not commit secrets, face photos, embeddings, or large model weights.

## Research clutter

See `liveness/RESEARCH.md`. Research FAS experiments are retained for the thesis; they are not used for ACCEPT.

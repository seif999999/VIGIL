# Friend-pipeline integration report

**Branch:** `friend-pipeline` (local only — not pushed)  
**Baseline `main`:** `eb7856bee44cb9ce8036b2e6644fe2ef70b63174` (unchanged)

## Architecture

```
Trusted capture → HMAC → SCRFD (1 face)
  → DeepFace FasNet (full frame + SCRFD xywh; native 2.7/4.0 crops)
  → FaceNet-128 vs local JSON (euclidean < 9.0 UNVALIDATED)
  → ACCEPT iff REAL ∧ recognized
```

Legacy mn3 path preserved: `--enable-liveness` / `--enable-visual-baseline`.

## Adopted (mechanisms via DeepFace API — not friend’s private source)

- FasNet MiniFASNetV2@2.7 + MiniFASNetV1SE@4.0 ensemble semantics
- FaceNet-128 + euclidean match threshold 9.0 (explicitly unvalidated)
- Decision rule REAL+recognized → ACCEPT; spoof/unknown/0/multi face → DENY

## Preserved from VIGIL

- HMAC protocol, session/counter, fail-closed reject
- Trusted Windows/webcam capture
- SCRFD + strict face-count gate
- anti-spoof-mn3 + provisional demo policy
- Existing automated security tests

## Files added

| File | Role |
|------|------|
| `liveness/fasnet_liveness.py` | FasNet wrapper |
| `liveness/facenet_identity.py` | FaceNet + JSON enrollments |
| `liveness/auth_decision.py` | Secure ACCEPT rules |
| `liveness/auth_pipeline.py` | SCRFD→FasNet→FaceNet |
| `liveness/enroll_identity.py` | Enrollment CLI |
| `liveness/live_friend_auth.py` | Non-HMAC live demo / latency |
| `liveness/parity_fasnet_check.py` | Same-bbox FasNet parity |
| `liveness/test_auth_decision.py` | Decision unit tests |
| `liveness/test_friend_auth_gates.py` | FACE_OK / HMAC gate tests |
| `liveness/requirements-friend-auth.txt` | Deps |
| `liveness/RESEARCH.md` | Research index |
| `liveness/data/README.md` | Enrollment privacy note |
| `README.md`, `LIVE_PHYSICAL_TESTS.md`, `CLEANUP_APPROVAL_NEEDED.md`, `INTEGRATION_REPORT.md` | Docs |

## Files modified

- `hmac/verifier_server.py` — `--enable-friend-auth`, latency stats
- `.gitignore` — `liveness/data/` biometric store
- `liveness/README.md` — friend-auth section

## Compatibility / security notes

- Friend’s OpenCV detector bbox ≠ SCRFD bbox; FasNet still uses **native** scaled crops on the SCRFD box (correct API). Absolute scores may differ from friend’s OpenCV-only tester on the same scene — expected.
- Threshold 9.0 not calibrated on VIGIL media.
- No pickle enrollment loading; no employee ACL/GUI.
- Friend personal photos / `embeddings_deepface.pkl` not imported.

## Automated test results

| Suite | Result |
|-------|--------|
| `liveness` auth decision + gates + decision_policy | **17 OK** |
| `hmac/tests.py` | **12 OK** |
| FasNet parity (`parity_fasnet_check.py`, same bbox) | **is_real match, score Δ=0** |

SCRFD+FasNet+FaceNet end-to-end LIVE and integrated HMAC physical tests: **not run** (require user camera / enrollment).

## Latency

- Optimization applied: **one-time** `warm_up()` for FasNet+FaceNet; reuse across frames (no per-frame reload).
- Steady-state LIVE numbers: **pending** user `live_friend_auth.py` / verifier summary after physical tests.
- Friend harness reference (separate workspace): ~1.4–5 s including cold start on some samples — not claimed as VIGIL results.

## Cleanup

- Removed local `__pycache__/` trees (regenerable).
- Research code retained; see `CLEANUP_APPROVAL_NEEDED.md` for optional deletions.

## Commands (summary)

```text
# Enroll
cd liveness
python -m venv .venv_auth
.venv_auth\Scripts\pip install -r requirements-friend-auth.txt
.venv_auth\Scripts\python enroll_identity.py --identity YOU --camera 0

# HMAC LIVE
cd ..\hmac
python verifier_server.py --enable-friend-auth
# other terminal:
python trusted_capture.py
```

**No commit / push performed** (awaiting your approval).

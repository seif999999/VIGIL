# Cleanup items requiring approval (do not delete yet)

Ambiguous or irreversible candidates. **Not deleted** pending your OK.

## A. Keep (essential) — no action

- `hmac/protocol.py`, `trusted_capture.py`, `verifier_server.py`, `keygen.py`, `tests.py`, …
- `face_detection/detector.py`, tests, `MODEL_INFO.txt`
- `liveness/detector.py` (mn3), `pipeline.py`, `crop.py`, `decision_policy.py`, `face_bridge.py`, …
- New friend-auth: `fasnet_liveness.py`, `facenet_identity.py`, `auth_*.py`, `enroll_identity.py`, …

## B. Keep for thesis (research — propose leave in place)

- `liveness/research/**`
- `liveness/_silent_face_refs/`, `_cvpr2024_refs/`, `_iadg_refs/`
- `liveness/diagnose_*.py`
- Local (gitignored) `liveness/models/research/**`, diagnostic run folders

## C. Safe disposable (generated) — may remove on approval

Confirm these are not needed as thesis evidence before deleting:

- `**/__pycache__/`
- `liveness/windows_capture_bridge/obj/` (rebuildable)
- Empty or stale `hmac/replay_artifacts/`, `webots/run_logs/` if present
- `liveness/.venv_fsfm/` (research-only venv; large) — keep if still used for FLIP/FSFM

## D. Need your decision

| Item | Why ambiguous |
|------|----------------|
| Move `diagnose_*.py` + `research/` under `liveness/archive/` | Improves clarity but churns imports/paths |
| Delete `liveness/_iadg_refs/` etc. after thesis citation is done | Irreversible without re-download |
| Remove `hmac/attacker_proxy.py` | Useful for security demos; not production |
| Delete local Camo diagnostic frame trees | Private faces; already gitignored — delete only if you want disk back |

Reply with which rows in **C** / **D** to apply.

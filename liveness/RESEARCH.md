# Research / diagnostic archive index

Thesis-related experiments live alongside production modules. They are **not**
on the HMAC → SCRFD → FasNet/FaceNet acceptance path.

| Path | Role |
|------|------|
| `research/` | Independent FAS wrappers (FLIP, FSFM, IADG, MiniFAS, CVPR2024) |
| `_silent_face_refs/`, `_cvpr2024_refs/`, `_iadg_refs/` | Upstream reference snapshots |
| `diagnose_*.py`, `audit_liveness_accuracy.py` | Offline / Camo diagnostics |
| `models/research/` | Large checkpoints (gitignored) — keep locally for thesis; **back up externally before deleting** |
| `diagnostic_*`, `camera_domain_runs/`, `benchmark_results/` | Generated runs (gitignored) |
| `requirements-research-fsfm.txt` | Recreate PyTorch research venv (`.venv_fsfm`) if FSFM/FLIP diagnostics are needed |

Operational auth uses `.venv_auth` + `requirements-friend-auth.txt` only.

Legacy production liveness remains `detector.py` (anti-spoof-mn3).
Experimental acceptance path: `fasnet_liveness.py` + `facenet_identity.py` + `auth_pipeline.py`.

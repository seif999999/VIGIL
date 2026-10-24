# FasNet V1 threshold calibration (research)

**Status:** Collection + offline analysis only.  
**Default auth policy:** `shadow` — research candidate gates (0.85/0.85) are **diagnostic**, not activated.  
**Do not** treat FasNet `winning_score` as a calibrated probability.

Scores recorded:

| Field | Meaning |
|-------|---------|
| `fasnet_winning_score` | Ensemble winning-class confidence = `sum(softmax)[argmax]/2` |
| `fasnet_real_score_avg` | Average softmax mass on REAL (index 1) |
| `fasnet_spoof_score_combined` | Average mass on spoof classes (0+2) |
| `fasnet_native_class` | Argmax class (REAL/SPOOF) |

## Collect sessions (preserve all prior JSONL)

Use default ONNX FasNet (and ONNX FaceNet when exported). Shadow policy stays default.

```powershell
cd C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\liveness
$env:PYTHONIOENCODING='utf-8'
$py = ".\.venv_auth\Scripts\python.exe"

# REAL — live person, stable lighting
& $py live_friend_auth.py --camera 0 --record-label REAL
# Wait for preview + auth; press M to MARK; move naturally; M to pause; Q to end

# PRINT — paper photo attack
& $py live_friend_auth.py --camera 0 --record-label PRINT

# SCREEN — phone/monitor replay
& $py live_friend_auth.py --camera 0 --record-label SCREEN
```

- Output: `data/fasnet_calibration/session_<LABEL>_<UTC>.jsonl` (gitignored).  
- No images stored. Existing files are never overwritten.  
- Prefer **≥2 sessions per label** for a held-out validation split.

Force backends if needed:

```powershell
$env:VIGIL_FASNET_BACKEND='onnx'
$env:VIGIL_FACENET_BACKEND='onnx'
```

## Analyze (does not activate thresholds)

```powershell
& $py analyze_fasnet_calibration.py --dir data\fasnet_calibration --json-out data\fasnet_calibration\report_v1.json
```

Optional candidate list (symmetric + full asymmetric grid in JSON):

```powershell
& $py analyze_fasnet_calibration.py --candidates 0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95 --json-out data\fasnet_calibration\report_v1.json
```

## After review — activating validated gates (manual opt-in)

1. Confirm held-out validation is adequate and false-REAL on PRINT/SCREEN is acceptable.  
2. Record chosen `real_min` / `spoof_min` in the thesis note.  
3. Opt in only when ready:

```powershell
& $py live_friend_auth.py --camera 0 --fasnet-policy active_research --enable-challenge1
```

`active_research` still uses **unvalidated** research candidates unless you change code after review — do **not** assume 0.85 is final V1.

## Limitations

- Frames within a session are temporally dependent.  
- Session-level majority votes are reported separately.  
- Calibration without PRINT/SCREEN cannot support security claims.  
- Challenge 1 physical resistance is **not** proven by unit tests.

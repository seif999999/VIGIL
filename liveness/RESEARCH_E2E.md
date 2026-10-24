# LOCAL RESEARCH end-to-end authentication test

Integrated physical test for SCRFD + **frozen FasNet temporal PAD** + FaceNet + Challenge 1.

**Not production.** Default `live_friend_auth.py` is unchanged.

## Thesis rule (Challenge 1)

Challenge 1 triggers **only** when the joint assessment is UNCERTAIN — never when FasNet is confidently REAL and FaceNet is a confident MATCH.

## Stage readiness

| Stage | Status |
|-------|--------|
| SCRFD | Implemented |
| FasNet frozen PAD (5s median REAL@0.95 / SPOOF@0.85 / min_valid=5) | Implemented |
| FaceNet binary MATCH (L2 &lt; 9.0) / UNKNOWN (≥ 9.0) | Implemented |
| FaceNet identity UNCERTAIN zone | **NOT AVAILABLE** — no GENUINE/IMPOSTOR calibration evidence yet |
| Challenge 1 on PAD UNCERTAIN + MATCH | Implemented |
| Challenge 1 on REAL + MATCH | **Disabled** (thesis) |
| Challenge 2 voice | **NOT_IMPLEMENTED** |
| Default live auth | Unchanged |

## Supported joint policy

| PAD | Identity | Outcome |
|-----|----------|---------|
| REAL | MATCH | **ACCEPT** (direct) |
| UNCERTAIN | MATCH | **Challenge 1** |
| SPOOF | * | **DENY** |
| * | UNKNOWN (L2 ≥ 9.0) | **DENY** (binary rejection — not “UNCERTAIN”) |
| * | UNAVAILABLE | **DENY** |
| INVALID/ERROR | * | **RETRY** then fail closed |

After Challenge 1 gesture OK → fresh 5s PAD. Post-C1 REAL+MATCH → face-stage ACCEPT (research). Post-C1 still UNCERTAIN → Challenge 2 unavailable (fail closed). Gesture alone never ACCEPT.

## Challenge 1 (research E2E only)

- Exactly **one** randomized instruction (`LOOK_LEFT` / `LOOK_RIGHT` / `LOOK_UP` / `LOOK_DOWN`).
- Multi-step sequences disabled in this tool (production `live_friend_auth` unchanged).
- **Blocking TTS**: spoken line finishes, brief pause, then the move window opens.
- Default move window **18s** (`--c1-timeout-ms`); HUD shows LISTEN vs MOVE + countdown.

## Not yet implementable

Without a calibrated FaceNet gray zone:

- Confident REAL + identity UNCERTAIN → Challenge 1
- Both UNCERTAIN → Challenge 1

Collect distances first with `live_identity_calibration.py` (GENUINE / IMPOSTOR). Do not invent identity uncertainty thresholds.

## Known limitation

Moving printed photographs can sometimes fool FasNet and landmark head-motion checks. Integration does not eliminate this.

## Launch

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\liveness"; python live_research_e2e_auth.py --camera 0
```

HUD: left panel with live FasNet/FaceNet, joint decision, 5s PAD countdown + progress bar, window #, valid counts, C1/C2 status. **R**=reset, **Q**=quit, **D**=debug.

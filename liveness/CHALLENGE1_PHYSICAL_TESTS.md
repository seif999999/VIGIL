# Challenge 1 — manual physical test checklist

Unit tests do **not** prove physical attack resistance. Run these before any security claim.

## Setup

```powershell
cd C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\liveness
$env:PYTHONIOENCODING='utf-8'
.\.venv_auth\Scripts\python.exe live_friend_auth.py --camera 0 --fasnet-policy active_research --enable-challenge1
```

Enrolled identity must be present. Shadow-only mode will rarely enter Challenge 1.

## Genuine user

| # | Case | Expected |
|---|------|----------|
| G1 | Follow spoken LOOK LEFT / RIGHT / UP / DOWN | Gesture verifies; fresh REAL+recognized → ACCEPT or still UNCERTAIN → Challenge 2 required |
| G2 | Two-step sequence in order | Both steps required; order matters |
| G3 | Slow correct movement | Still passes if angles held |
| G4 | Correct move then return to center before hold ends | May fail hold / need retry |

## Fail-closed / spoof

| # | Case | Expected |
|---|------|----------|
| A1 | Static print / phone frozen frame | STATIC_NO_MOVE or TIMEOUT; no ACCEPT |
| A2 | Move print in wrong direction | WRONG_DIRECTION |
| A3 | Replay video of prior challenge | Must not ACCEPT without fresh FasNet/FaceNet REAL+match |
| A4 | Remove face mid-challenge | FACE_LOST |
| A5 | Second person enters frame | MULTIPLE_FACES |
| A6 | Cover camera | CAMERA_DISCONNECTED / NO_FACE |
| A7 | Unknown person completes gesture | DENY unknown identity |
| A8 | Screen spoof that FasNet marks SPOOF after gesture | DENY spoof (gesture cannot override) |
| A9 | Gesture OK but FasNet stays UNCERTAIN | CHALLENGE_REQUIRED / challenge_2_required — **not** ACCEPT |

## LEFT/RIGHT mirror check

Confirm HUD LOOK LEFT matches the user’s own left (subject-centric). If the preview is mirrored for display, labels must still match subject motion — fail the build if LEFT/RIGHT feel inverted.

## Pose threshold note

`PoseThresholds` are `EMPIRICAL_UNVALIDATED`. Adjust only after these physical tests; do not treat defaults as production-safe.

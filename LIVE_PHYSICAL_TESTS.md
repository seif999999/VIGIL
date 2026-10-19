# LIVE physical tests (user-operated)

Automated suites do **not** replace these. Run after enrollment.

## Prep

```text
cd liveness
.venv_auth\Scripts\python enroll_identity.py --identity <your_name> --camera 0 --count 5
```

## Integrated (HMAC) session

**Terminal A**

```text
cd hmac
python verifier_server.py --enable-friend-auth
```

**Terminal B**

```text
cd hmac
python trusted_capture.py
```

(Use the same Windows camera index / MediaCapture path you already validated for trusted capture.)

## Scenarios to record

| # | Scenario | Expected |
|---|----------|----------|
| 1 | Genuine, stationary | ACCEPT when enrolled |
| 2 | Genuine, head turn | ACCEPT or brief DENY/UNCERTAIN — log FasNet/FaceNet distances |
| 3 | Genuine with / without glasses | Compare distances vs threshold 9.0 |
| 4 | PRINT attack | DENY spoof |
| 5 | SCREEN attack | DENY spoof |
| 6 | Unknown consenting person | DENY unknown_identity |

Log median SCRFD / FasNet / FaceNet / total ms from verifier summary.

## Non-HMAC latency profiling

```text
cd liveness
.venv_auth\Scripts\python live_friend_auth.py --camera 0
```

Press SPACE for scored samples; note median after warm-up (ignore first sample).

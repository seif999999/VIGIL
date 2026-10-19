# VIGIL Stage 1 — HMAC frame authentication

This module is the first practical layer of the VIGIL bachelor thesis project. It proves that **camera frames can be wrapped in a signed binary package** and verified **before** any AI pipeline (FAS, EdgeFace, etc.) would see them.

Stage 1 uses **real JPEG file bytes** only. There is no webcam, network transport, or attacker proxy yet.

## Purpose

Every frame that enters the trusted processing path must:

1. Match the expected **binary format**
2. Pass **HMAC-SHA256** verification (integrity + authenticity of the signer)
3. Belong to the current **session**
4. Arrive with the expected **monotonic frame counter** (replay / reorder protection)

If any check fails, the frame is rejected and must not be decoded or processed.

## Threat model (Version 1)

**Assumptions**

- The 256-bit HMAC secret is known only to the frame signer and the verifier.
- An attacker may modify bytes in transit, replace the image, replay old frames, or inject unsigned data.
- An attacker does **not** know the secret key (otherwise they could forge valid tags).

**HMAC protects against**

- Changing JPEG payload bytes
- Swapping in a different image
- Tampering with version, session ID, counter, or declared JPEG length
- Injecting frames without a valid tag (when the key is secret)

**Frame counter protects against**

- Sending the same authenticated package twice (duplicate)
- Replaying an older frame after newer ones were accepted
- Skipping ahead (out-of-order counters)

**Session ID protects against**

- Reusing a **legitimate, correctly signed** frame from a **previous** authentication session in a **new** session (HMAC still verifies; session check fails)

**Known limitation**

HMAC does **not** prove that pixels physically came from a specific camera sensor. It proves the package was built by a party holding the key and was not altered afterward. Proving sensor origin is outside Stage 1 and is part of the wider VIGIL threat model.

## Binary package format

All multi-byte integers are **big-endian** (network byte order).

| Field           | Size     | Description                          |
|----------------|----------|--------------------------------------|
| `VERSION`      | 1 byte   | `0x01` for protocol version 1        |
| `SESSION_ID`   | 16 bytes | Random 128-bit session identifier    |
| `FRAME_COUNTER`| 8 bytes  | Unsigned 64-bit monotonic counter    |
| `JPEG_LENGTH`  | 4 bytes  | Unsigned 32-bit length of JPEG body  |
| `JPEG_BYTES`   | variable | Raw JPEG file bytes                  |
| `HMAC_TAG`     | 32 bytes | HMAC-SHA256 output                   |

**Header length:** 29 bytes  

**Full package:** `HEADER || JPEG_BYTES || HMAC_TAG`

## HMAC input

The tag is computed over **header + JPEG bytes only**:

```text
tag = HMAC-SHA256(secret_key, header || jpeg_bytes)
```

The tag itself is **not** included in the HMAC input. Everything in the header and the full JPEG payload is authenticated.

## Why HMAC-SHA256?

HMAC-SHA256 is a standard message authentication code: fast, widely understood, and available in the Python standard library (`hmac` + `hashlib`). It provides integrity and authenticity under a shared secret without custom cryptography.

## Why session ID?

When a user starts a new authenticated session (new login, new stream, etc.), the verifier expects a fresh `SESSION_ID`. Frames signed under an old session remain cryptographically valid but are **rejected** at the session check, so they cannot be replayed across sessions.

## Why frame counter?

HMAC alone cannot detect **replay** of an unmodified package. A monotonic counter per session ensures each accepted frame number is seen exactly once in order (starting at `0` for a new verifier).

## Why no timestamp (Version 1)?

Version 1 deliberately uses **session ID + counter** for freshness. Timestamps add clock sync and policy complexity; they can be added in a later protocol version if needed.

## Generate the secret key

From this directory:

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\hmac"
python keygen.py
```

The key is written to `secrets\hmac.key` (32 raw bytes). Existing keys are **not** overwritten unless you pass `--force`. The key is never printed.

**Do not commit `secrets/`** — it is listed in `.gitignore`.

## Test images

Place **real captured** JPEGs in `test_images\`:

- `original_1.jpg` (required)
- `fake_image.jpg` (required)
- `original_2.jpg` (optional)

Do not resize or recompress them for testing; HMAC is over the **exact file bytes**. See `test_images\README.txt`.

## Run tests

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\hmac"
python keygen.py
python -m unittest -v tests.py
```

Or:

```powershell
python tests.py
```

The runner prints a short PASS/FAIL/SKIP summary for thesis-friendly labels.

If images are missing, image-dependent tests are **skipped** until you add the files.

## Verification order (`FrameVerifier`)

1. Structural parse (length, version, bounds)
2. Recompute HMAC
3. Constant-time compare (`hmac.compare_digest`)
4. Session ID match
5. Frame counter (duplicate / replay / out-of-order)
6. Accept and advance expected counter

Rejected frames do **not** advance the counter. JPEG bytes are not decoded before authentication succeeds.

## Stage 2 — Live HMAC-verified webcam (local)

Stage 2 connects the same `protocol.py` logic to your laptop webcam. Capture, signing, and verification still run **inside one local process**; networking and attacker simulation are later stages.

**Dependency:** OpenCV (`opencv-python`) for capture and JPEG encode/decode.

**Run:**

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\hmac"
python live_camera.py
```

Optional camera index (default `0`):

```powershell
python live_camera.py --camera 1
```

Optional auto-stop after N seconds (otherwise use Q or ESC):

```powershell
python live_camera.py --duration 30
```

**Flow:**

```text
webcam → JPEG encode (in memory) → HMAC package (session + counter)
      → FrameVerifier → ACCEPT only → decode verified JPEG → display
```

The window title is **VIGIL - HMAC Verified Camera**. Overlays show session prefix, frame number, and `HMAC: VERIFIED`. Press **Q** or **ESC** to exit; a short summary prints frame counts and approximate FPS.

Raw webcam frames are **not** shown directly—only JPEG bytes taken from an authenticated package after verification succeeds.

## Stage 3 — TCP split: trusted capture and verifier (local)

Stage 3 separates signing and verification into **two processes** on localhost. TCP carries bytes only; **HMAC** still provides authentication and integrity.

| Program | Role |
|---------|------|
| `verifier_server.py` | Creates session ID, listens on TCP, verifies packages, displays **only** accepted JPEGs |
| `trusted_capture.py` | Opens webcam, signs frames with the **verifier-assigned** session, sends packages |

**Flow:**

```text
trusted_capture.py                    verifier_server.py
  webcam → JPEG → sign                     listen TCP
       → length prefix + package  ──TCP──→  recv → FrameVerifier
                                            ACCEPT → decode → display
```

**Session handshake:** On connect, the verifier sends **16 raw bytes** (session ID). Capture must **not** generate its own session. Both sides load the same `secrets\hmac.key` (capture signs; verifier verifies).

**TCP framing (transport only, not part of HMAC):**

```text
4-byte big-endian unsigned length  ||  complete HMAC package
```

The HMAC package format is unchanged from Stage 1.

**Terminal 1 — verifier (start first):**

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\hmac"
python verifier_server.py
```

Optional: `--host 127.0.0.1 --port 5000`

**Terminal 2 — capture:**

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\hmac"
python trusted_capture.py --duration 25
```

Optional: `--host`, `--port`, `--camera 1`, `--duration`

Capture **signs only** (no local `FrameVerifier`). Press **Q** or **ESC** on the verifier window to stop display; capture can use `--duration` or Ctrl+C.

Stage 2 (`live_camera.py`) remains a single-process demo and is unchanged in purpose.

## Stage 4 — Attacker proxy (digital path only)

Stage 4 inserts **`attacker_proxy.py`** between capture and verifier. The proxy has **no HMAC secret** — it only manipulates bytes on localhost TCP.

```text
trusted_capture.py  →  :5001  →  attacker_proxy.py  →  :5000  →  verifier_server.py
```

- **Verifier** still creates the session; the proxy **forwards** the 16-byte session ID unchanged.
- **TCP framing** is unchanged: 4-byte big-endian length + HMAC package.
- **HMAC** detects tampering, replacement, and fake injection (no valid tag without the key).
- **Frame counter** rejects duplicate / old-frame replay within a session (HMAC may still be valid).
- **Fresh session ID** per verifier run rejects **old-session replay** (`WRONG_SESSION`).
- HMAC does **not** prove physical camera origin (USB/virtual-camera attacks before signing are out of scope).

**V1 fail-closed authentication:** Verifier uses **strict sequential counters** (no gap resync). On any `FrameVerifier` security rejection (`HMAC_MISMATCH`, `WRONG_SESSION`, `REPLAY_OR_DUPLICATE`, `OUT_OF_ORDER`, etc.), the verifier **aborts that authentication session**: logs `[SECURITY] Authentication stream rejected; terminating current session.`, closes the TCP connection, and reports **`Result: SECURITY_REJECT`**. A new attempt requires restarting the verifier (fresh session ID) and reconnecting capture. VIGIL does **not** try to resynchronize after a detected stream attack.

**Default ports:** verifier `5000`, proxy listen `5001`, capture connects to `5001`.

### TEST A — Transparent proxy

```powershell
# Terminal 1
python verifier_server.py --port 5000

# Terminal 2
python attacker_proxy.py --listen-port 5001 --verifier-port 5000 --mode normal

# Terminal 3
python trusted_capture.py --port 5001 --duration 15
```

Expected: all frames verify, **0 rejected**, verifier **`Result: PASS`**.

Direct Stage 3 (no proxy): `trusted_capture.py --port 5000` → verifier must still show **PASS** with zero rejects.

### TEST B — JPEG tampering

```powershell
python attacker_proxy.py --listen-port 5001 --verifier-port 5000 --mode tamper-jpeg --attack-after 20
python trusted_capture.py --port 5001 --duration 15
```

Expected: one **`HMAC_MISMATCH`**, tampered frame not displayed, session **`SECURITY_REJECT`** (no counter resync).

### TEST C — JPEG replacement

```powershell
python attacker_proxy.py --mode replace-jpeg --attack-after 20 --replacement-image test_images\original_2.jpg
```

Use any **existing** JPEG path different from the live signed bytes. Expected: **`HMAC_MISMATCH`**, session **`SECURITY_REJECT`**.

### TEST D — Fake digital injection

```powershell
python attacker_proxy.py --mode inject-fake --attack-after 20
```

Expected: fake → **`HMAC_MISMATCH`**, session **`SECURITY_REJECT`** (genuine same-counter frame may not be processed if session already closed).

### TEST E — Duplicate

```powershell
python attacker_proxy.py --mode duplicate --attack-after 20
```

Expected: first **ACCEPT**, duplicate **`REPLAY_OR_DUPLICATE`**, session **`SECURITY_REJECT`**.

### TEST F — Old-frame replay

```powershell
python attacker_proxy.py --mode old-frame-replay --attack-after 20
```

Expected: replay → **`REPLAY_OR_DUPLICATE`**, session **`SECURITY_REJECT`**.

### TEST G — Out-of-order

```powershell
python attacker_proxy.py --mode out-of-order --attack-after 20
```

Expected: early N+1 → **`OUT_OF_ORDER`**, session **`SECURITY_REJECT`**.

### TEST H — Old-session replay (two runs)

**Run 1 — record:**

```powershell
python verifier_server.py --port 5000
python attacker_proxy.py --listen-port 5001 --verifier-port 5000 --mode record-old-session --replay-file replay_artifacts\old_session_package.bin
python trusted_capture.py --port 5001 --duration 10
```

Stop all. **Run 2 — replay:**

```powershell
python verifier_server.py --port 5000
python attacker_proxy.py --listen-port 5001 --verifier-port 5000 --mode old-session-replay --replay-file replay_artifacts\old_session_package.bin
python trusted_capture.py --port 5001 --duration 10
```

Expected: saved Session A package → **`WRONG_SESSION`**, session **`SECURITY_REJECT`**; restart verifier/capture for Session B traffic.

Stage 3 direct (`trusted_capture` → `verifier` on port 5000, no proxy) remains supported.

## Stage 5 — HMAC performance benchmark (measure only)

`benchmark_hmac.py` measures **security and localhost transport overhead** without changing the protocol or optimising code.

**What is measured**

- **Microbenchmark:** `create_frame_package()` and `FrameVerifier.verify()` on fixed in-memory JPEG bytes (no disk I/O in the timed loop, no OpenCV encode in crypto timings).
- **Package overhead:** bytes added by header + HMAC tag vs raw JPEG and vs baseline TCP record (`4-byte length + JPEG`).
- **Localhost TCP:** same JPEG sequence over `127.0.0.1` — baseline (`length + JPEG`) vs secured (`length + HMAC package`), no GUI/decode.

**What is NOT measured**

- WAN latency, real network paths, or live webcam FPS during the timed sections (optional `--camera` collects JPEGs first, then benchmarks offline).

**Method:** `time.perf_counter_ns()`, ≥100 crypto warm-up ops, configurable iterations (default 1000), multiple TCP trials (default 5×500 frames). Results print to the console and save as `benchmark_results\stage5_<timestamp>.json` (gitignored).

**Run (file-based):**

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\hmac"
python benchmark_hmac.py --image test_images\original_1.jpg --iterations 1000 --tcp-frames 500 --tcp-trials 5
```

**Optional real JPEG sizes from webcam (capture first, then benchmark):**

```powershell
python benchmark_hmac.py --camera 0 --capture-frames 300 --iterations 1000 --tcp-frames 500 --tcp-trials 5
```

Distinguish **security-processing throughput** (crypto only) from **localhost TCP FPS** from **full live-camera FPS** in Stage 2/3.

## Next stage (not implemented here)

Planned follow-up: FAS, EdgeFace, challenge logic, and full VIGIL fusion. Review Stage 5 numbers before proceeding.

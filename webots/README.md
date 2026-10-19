# VIGIL Webots integration (Baseline V1)

## Webots installation (required)

**Webots was not detected** on the machine used for initial integration (`webots` not in PATH; no install under `C:\Program Files\Webots`).

Install from [Cyberbotics Webots](https://cyberbotics.com/) (Windows installer). After install, confirm:

```powershell
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --version
```

Use a **R2023b or newer** build matching the world header in `worlds/vigil_indoor.wbt`.

Official references:

- [Python controllers](https://cyberbotics.com/doc/guide/controller-programming)
- [Camera device](https://cyberbotics.com/doc/reference/camera) — `Camera.getImage()` returns BGRA bytes; converted to BGR for VIGIL.

## Simulation disclaimer

The poster target uses **`textures/face_target.jpg`** (copy from `hmac/test_images/`). It is a **rendered image in simulation**, not a live human. Anti-spoof scores are **pipeline demos only**, not validation of physical liveness.

## Architecture

```
Webots Camera (controller vigil_camera)
    → TCP :5010 BGR frames (frame_protocol.py)
    → trusted_webots_capture.py (HMAC sign via trusted_capture.py)
    → verifier_server.py --enable-visual-baseline
    → SCRFD → anti-spoof-mn3 → provisional policy (0.85)
```

Desktop/phone capture is unchanged (`trusted_capture.py` with `--camera` or `--windows-device`).

## Setup texture (SCRFD-detectable rendered poster)

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\webots\textures"
copy "..\..\face_detection\my_face.jpeg" face_target.jpg
python ..\diagnose_poster_texture.py
```

Expect `SCRFD faces: 1` / `FACE_OK` on the **file** (simulation texture only).

## Camera diagnostic (black feed vs missing face)

First Webots frame saved from the bridge:

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\webots"
python trusted_webots_capture.py --save-diagnostic diagnostics\webots_bridge_frame.png --duration 30
```

Start Webots **after** the bridge is listening. The report includes dimensions, mean BGR, black/uniform flags, and SCRFD on that PNG.

Optional controller-side save (set before Play):

```powershell
$env:VIGIL_WEBOTS_DIAG_DIR = "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\webots\diagnostics"
```

## Run (three terminals)

**1 — Verifier (start first)**

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\hmac"
python verifier_server.py --enable-visual-baseline
```

**2 — Webots bridge (before pressing Play in Webots)**

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\webots"
python trusted_webots_capture.py --duration 60
```

**3 — Webots**

Open `webots/worlds/vigil_indoor.wbt` in Webots (File → Open World), or:

```powershell
& "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" `
  "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\webots\worlds\vigil_indoor.wbt"
```

Press **Play**. The `vigil_camera` controller connects to `127.0.0.1:5010`.

### Webots reset / restart (after world or controller changes)

1. Press **Stop** (or pause) in Webots.
2. **Reload World** (File → Reload World) or close and reopen `vigil_indoor.wbt`.
3. Restart **verifier** (Terminal 1) if the previous session ended with `SECURITY_REJECT`.
4. Start **bridge** (Terminal 2) and wait for `listening on 127.0.0.1:5010`.
5. Press **Play** in Webots (controller reconnects each run).
6. Check robot **Display** preview and verifier window for the poster; use `--save-diagnostic` if still dark.

Optional env vars for the controller: `VIGIL_WEBOTS_BRIDGE_HOST`, `VIGIL_WEBOTS_BRIDGE_PORT`.

## Automated tests (no Webots GUI)

```powershell
cd "C:\Users\smnam\OneDrive\Desktop\VIGIL IMPL\webots"
python webots_bridge_test.py
cd ..\hmac
python integrated_visual_baseline_test.py
python tests.py
```

# VIGIL Webots scene geometry (manual verification)

Webots uses **Y-up**, **X-right**, **Z-forward** in world coordinates.

## Camera (official convention)

The `Camera` node looks along **−Z** in its **local** frame ([Camera reference](https://cyberbotics.com/doc/reference/camera)).

## Current layout (`vigil_indoor.wbt`)

| Item | World position (center) | Notes |
|------|-------------------------|--------|
| Robot base | `(0, 0.06, 0.74)` | Identity rotation |
| Camera (child Transform) | offset `(0, 0.38, 0)` → ≈ `(0, 0.44, 0.74)` | **No extra rotation** → looks world **−Z** |
| Poster plane center | `(0, 0.44, 0.36)` | `rotation 0 1 0 π` → textured face normal **−Z** (toward robot) |
| Poster size | `0.62 × 0.82` m (Plane width × height) | Upright portrait in world **XY** plane |

**Line of sight:** camera at `z ≈ 0.74` looks toward decreasing **Z**; poster at `z = 0.36` is **0.38 m** in front along **−Z**. Camera and poster centers share **Y ≈ 0.44 m** (horizontal aim).

**Why the old scene failed:** robot at `z = 0.08` with camera `rotation 0 1 0 π` often views **+Z** (sky blue + floor gray) while the poster sat at `z = 0.72` **behind** the robot relative to a **−Z** view.

## Quick GUI check (no VIGIL)

1. Open `worlds/vigil_indoor.wbt`.
2. In the 3D view, select viewpoint **`poster_alignment_check`** (Viewpoint list).
3. You should see the **rendered face poster** filling much of the view (simulation texture only).
4. Press Play briefly and confirm the robot **Display** shows the poster, not only blue/gray.

## After reload

Use the same bridge/HMAC commands as `README.md`. Save a diagnostic frame:

```powershell
python trusted_webots_capture.py --save-diagnostic diagnostics\webots_bridge_frame.png --duration 30
```

Expect non-uniform pixels and SCRFD `FACE_OK` on the PNG when the poster is visible (not proof of physical liveness).

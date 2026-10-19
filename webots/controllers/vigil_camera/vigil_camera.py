"""Webots robot controller: Camera.getImage() → VIGIL bridge (127.0.0.1:5010)."""

from __future__ import annotations

import os
import socket
import struct
from pathlib import Path

from controller import Camera, Display, Robot

BRIDGE_HOST = os.environ.get("VIGIL_WEBOTS_BRIDGE_HOST", "127.0.0.1")
BRIDGE_PORT = int(os.environ.get("VIGIL_WEBOTS_BRIDGE_PORT", "5010"))
_MAGIC = b"VGBR"
_HEADER = struct.Struct(">4sII")


def _bgra_bytes_to_bgr(data: bytes, width: int, height: int) -> bytes:
    out = bytearray(width * height * 3)
    j = 0
    stride = width * 4
    for row in range(height):
        base = row * stride
        for col in range(width):
            i = base + col * 4
            out[j] = data[i]
            out[j + 1] = data[i + 1]
            out[j + 2] = data[i + 2]
            j += 3
    return bytes(out)


def _send_bgr(sock: socket.socket, bgr: bytes, width: int, height: int) -> None:
    sock.sendall(_HEADER.pack(_MAGIC, width, height) + bgr)


def _connect_bridge() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((BRIDGE_HOST, BRIDGE_PORT))
    return sock


def _maybe_save_controller_diagnostic(
    bgr: bytes, width: int, height: int, frame_index: int
) -> None:
    diag_dir = os.environ.get("VIGIL_WEBOTS_DIAG_DIR")
    if not diag_dir or frame_index != 5:
        return
    try:
        import numpy as np

        arr = np.frombuffer(bgr, dtype=np.uint8).reshape((height, width, 3))
        out = Path(diag_dir) / "webots_controller_frame.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        # Webots controller may not have cv2; write minimal PPM if needed
        try:
            import cv2

            cv2.imwrite(str(out), arr)
        except ImportError:
            ppm = out.with_suffix(".ppm")
            with ppm.open("wb") as f:
                f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
                f.write(arr[:, :, ::-1].tobytes())
            out = ppm
        mean = float(arr.mean())
        print(
            f"vigil_camera diagnostic frame {frame_index}: {width}x{height} "
            f"mean={mean:.2f} saved={out}"
        )
    except Exception as exc:
        print(f"vigil_camera diagnostic save failed: {exc}")


def main() -> None:
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    camera = Camera("camera")
    camera.enable(timestep)
    display = Display("display")
    display.setOpacity(1.0)

    bridge = _connect_bridge()
    print(
        f"vigil_camera: streaming to {BRIDGE_HOST}:{BRIDGE_PORT} "
        "(rendered poster target — not live human liveness ground truth)"
    )

    frame_index = 0
    try:
        while robot.step(timestep) != -1:
            data = camera.getImage()
            if not data:
                continue
            width = camera.getWidth()
            height = camera.getHeight()
            if isinstance(data, str):
                data = data.encode("latin-1")
            bgr = _bgra_bytes_to_bgr(data, width, height)
            try:
                display.imageNew(data, Display.RGBA, width, height)
            except (AttributeError, TypeError):
                pass
            _maybe_save_controller_diagnostic(bgr, width, height, frame_index)
            frame_index += 1
            try:
                _send_bgr(bridge, bgr, width, height)
            except OSError as exc:
                print(f"vigil_camera: bridge send failed: {exc}")
                break
    finally:
        bridge.close()


if __name__ == "__main__":
    main()

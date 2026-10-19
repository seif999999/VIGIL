"""FFmpeg DirectShow named-device capture → BGR numpy frames."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

DEFAULT_FPS = 30


@dataclass(frozen=True)
class DshowCaptureInfo:
    device_name: str
    requested_width: int
    requested_height: int
    requested_fps: int
    ffmpeg_command: str


def print_dshow_capture_info(info: DshowCaptureInfo, *, actual_wh: tuple[int, int] | None) -> None:
    print(f"Capture source: {info.device_name}")
    print("Capture method: FFmpeg DirectShow")
    print(f"Resolution: {info.requested_width}x{info.requested_height}")
    print(f"FPS: {info.requested_fps}")
    if actual_wh:
        print(f"Actual frame: {actual_wh[0]}x{actual_wh[1]}")
    print(f"FFmpeg command: {info.ffmpeg_command}")


def _build_ffmpeg_command(
    device_name: str,
    width: int,
    height: int,
    fps: int,
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install FFmpeg and ensure ffplay/ffmpeg work."
        )
    input_spec = f"video={device_name}"
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "dshow",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        input_spec,
        "-an",
        "-sn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]


class FfmpegDshowCapture:
    """Read BGR frames from a Windows DirectShow device via FFmpeg stdout."""

    def __init__(
        self,
        device_name: str,
        *,
        width: int,
        height: int,
        fps: int = DEFAULT_FPS,
    ) -> None:
        self._device_name = device_name
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_bytes = width * height * 3
        self._cmd = _build_ffmpeg_command(device_name, width, height, fps)
        self._proc: subprocess.Popen[bytes] | None = None
        self._last_frame_shape: tuple[int, int] | None = None

    @property
    def capture_info(self) -> DshowCaptureInfo:
        return DshowCaptureInfo(
            device_name=self._device_name,
            requested_width=self._width,
            requested_height=self._height,
            requested_fps=self._fps,
            ffmpeg_command=" ".join(_quote_arg(a) for a in self._cmd),
        )

    @property
    def last_frame_size(self) -> tuple[int, int] | None:
        return self._last_frame_shape

    def start(self) -> None:
        if self._proc is not None:
            return
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise RuntimeError(f"Failed to start FFmpeg: {exc}") from exc
        time.sleep(1.0)
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"FFmpeg exited immediately (code {self._proc.returncode}). "
                "Check device name and that the virtual camera is running."
            )

    def _read_exact(self, n: int) -> bytes | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        chunks: list[bytes] = []
        remaining = n
        while remaining > 0:
            if self._proc.poll() is not None:
                return None
            chunk = self._proc.stdout.read(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._proc is None:
            raise RuntimeError("Call start() before read()")
        raw = self._read_exact(self._frame_bytes)
        if raw is None or len(raw) != self._frame_bytes:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )
        frame = frame.copy()
        self._last_frame_shape = (self._width, self._height)
        return True, frame

    def release(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)


def _quote_arg(arg: str) -> str:
    if " " in arg or '"' in arg:
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def open_dshow_capture(
    device_name: str,
    *,
    width: int,
    height: int,
    fps: int = DEFAULT_FPS,
) -> FfmpegDshowCapture:
    cap = FfmpegDshowCapture(
        device_name, width=width, height=height, fps=fps
    )
    cap.start()
    return cap


def _probe_main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a few FFmpeg DirectShow BGR frames (connectivity test)",
    )
    parser.add_argument(
        "--dshow-device",
        required=True,
        help='e.g. "Nazmy (Windows Virtual Camera)"',
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    cap = open_dshow_capture(
        args.dshow_device,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print_dshow_capture_info(cap.capture_info, actual_wh=None)
    ok_count = 0
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if ok and frame is not None:
                ok_count += 1
                h, w = frame.shape[:2]
                print(f"Frame {i + 1}: OK shape=({h}, {w}, 3)")
            else:
                print(f"Frame {i + 1}: FAIL")
        wh = cap.last_frame_size
        if wh:
            print(f"\nActual resolution from frame.shape: {wh[0]}x{wh[1]}")
        print(f"Valid frames: {ok_count}/{args.frames}")
    finally:
        cap.release()
    sys.exit(0 if ok_count > 0 else 1)


if __name__ == "__main__":
    _probe_main()

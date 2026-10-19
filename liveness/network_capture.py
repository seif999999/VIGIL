"""Live network camera stream (HTTP/MJPEG, RTSP) → BGR frames via OpenCV."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NetworkCaptureInfo:
    stream_url: str
    actual_width: int
    actual_height: int
    reported_fps: float | None


def print_network_capture_info(info: NetworkCaptureInfo) -> None:
    print("Capture method: Network stream")
    print(f"Stream URL: {info.stream_url}")
    print(f"Actual frame: {info.actual_width}x{info.actual_height}")
    if info.reported_fps is not None:
        print(f"FPS (OpenCV property): {info.reported_fps:.3f}")


def _open_videocapture(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url)
    return cap


def _read_first_frame(
    cap: cv2.VideoCapture,
    *,
    timeout_s: float = 15.0,
) -> tuple[bool, np.ndarray | None]:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(0.05)
    return False, None


class NetworkStreamCapture:
    """read()/release() compatible with other VIGIL capture helpers."""

    def __init__(
        self,
        stream_url: str,
        *,
        connect_timeout_s: float = 15.0,
    ) -> None:
        self._url = stream_url
        self._cap = _open_videocapture(stream_url)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open network stream: {stream_url}"
            )
        ok, frame = _read_first_frame(
            self._cap, timeout_s=connect_timeout_s
        )
        if not ok or frame is None:
            self._cap.release()
            raise RuntimeError(
                f"Stream opened but no valid frame within {connect_timeout_s}s: "
                f"{stream_url}"
            )
        self._pending: np.ndarray | None = frame
        h, w = frame.shape[:2]
        fps_prop = float(self._cap.get(cv2.CAP_PROP_FPS))
        reported = fps_prop if fps_prop > 1.0 else None
        self._info = NetworkCaptureInfo(stream_url, w, h, reported)

    @property
    def capture_info(self) -> NetworkCaptureInfo:
        return self._info

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._pending is not None:
            frame = self._pending
            self._pending = None
            return True, frame
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def open_network_stream(
    stream_url: str,
    *,
    connect_timeout_s: float = 15.0,
) -> NetworkStreamCapture:
    return NetworkStreamCapture(
        stream_url, connect_timeout_s=connect_timeout_s
    )


def _probe_main() -> None:
    parser = argparse.ArgumentParser(
        description="Network stream connectivity test (no SCRFD/liveness)",
    )
    parser.add_argument("--stream-url", required=True)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--connect-timeout", type=float, default=15.0)
    args = parser.parse_args()

    try:
        cap = open_network_stream(
            args.stream_url,
            connect_timeout_s=args.connect_timeout,
        )
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    print_network_capture_info(cap.capture_info)
    ok_count = 0
    t0 = time.perf_counter()
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if ok and frame is not None:
                ok_count += 1
                h, w = frame.shape[:2]
                print(f"Frame {i + 1}: OK {w}x{h}")
            else:
                print(f"Frame {i + 1}: FAIL")
        elapsed = time.perf_counter() - t0
        if ok_count > 1 and elapsed > 0:
            print(f"Measured receive rate: {ok_count / elapsed:.2f} frames/s")
    finally:
        cap.release()

    print(f"Valid frames: {ok_count}/{args.frames}")
    sys.exit(0 if ok_count > 0 else 1)


if __name__ == "__main__":
    _probe_main()

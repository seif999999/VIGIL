"""Webcam open + resolution request/verify for liveness tools only."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
FALLBACK_WIDTH = 640
FALLBACK_HEIGHT = 480


@dataclass(frozen=True)
class WebcamCaptureInfo:
    requested_width: int
    requested_height: int
    actual_width: int
    actual_height: int
    used_fallback: bool


def _open_capture(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    return cap


def _measure_frame_size(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Read frames until one arrives; return (width, height) from pixel data."""
    for _ in range(8):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            return int(w), int(h)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return w, h


def _apply_resolution(cap: cv2.VideoCapture, width: int, height: int) -> tuple[int, int]:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return _measure_frame_size(cap)


def open_webcam(
    index: int,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> tuple[cv2.VideoCapture, WebcamCaptureInfo]:
    """
    Open camera, request width x height, fall back to 640x480 when the default
    1280x720 request is not honored by the device.
    """
    cap = _open_capture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")

    requested_w, requested_h = width, height
    actual_w, actual_h = _apply_resolution(cap, requested_w, requested_h)
    used_fallback = False

    if (
        requested_w == DEFAULT_WIDTH
        and requested_h == DEFAULT_HEIGHT
        and (actual_w < DEFAULT_WIDTH or actual_h < DEFAULT_HEIGHT)
    ):
        used_fallback = True
        cap.release()
        cap = _open_capture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen camera index {index}")
        actual_w, actual_h = _apply_resolution(cap, FALLBACK_WIDTH, FALLBACK_HEIGHT)

    info = WebcamCaptureInfo(
        requested_width=requested_w,
        requested_height=requested_h,
        actual_width=actual_w,
        actual_height=actual_h,
        used_fallback=used_fallback,
    )
    return cap, info


def print_resolution_info(info: WebcamCaptureInfo) -> None:
    print(
        f"Requested resolution: {info.requested_width}x{info.requested_height}"
    )
    print(f"Actual resolution: {info.actual_width}x{info.actual_height}")
    if info.used_fallback:
        print(
            f"[INFO] Camera did not provide {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}; "
            f"applied fallback {FALLBACK_WIDTH}x{FALLBACK_HEIGHT}."
        )

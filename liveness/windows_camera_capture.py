"""Native Windows MediaCapture / MediaFrameReader → BGR numpy (WinRT via Python)."""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

DEFAULT_DEVICE = "Nazmy (Windows Virtual Camera)"


@dataclass(frozen=True)
class WinRTCaptureInfo:
    device_name: str
    device_id: str
    source_group_id: str
    actual_width: int
    actual_height: int
    pixel_format: str


def print_winrt_capture_info(info: WinRTCaptureInfo) -> None:
    print("Capture method: Windows MediaCapture")
    print(f"Device: {info.device_name}")
    print(f"Actual resolution: {info.actual_width}x{info.actual_height}")
    print(f"Pixel format: {info.pixel_format}")
    print(f"Device ID: {info.device_id}")
    print(f"Source group ID: {info.source_group_id}")


async def _pick_source_group(display_name: str):
    from winrt.windows.media.capture.frames import MediaFrameSourceGroup

    groups = await MediaFrameSourceGroup.find_all_async()
    for grp in groups:
        if grp.display_name == display_name:
            return grp
    names = [g.display_name for g in groups]
    raise RuntimeError(
        f"Windows device not found: {display_name!r}. Available: {names}"
    )


async def _initialize_reader(display_name: str):
    from winrt.windows.media.capture import (
        MediaCapture,
        MediaCaptureInitializationSettings,
        MediaCaptureMemoryPreference,
    )
    from winrt.windows.media.capture.frames import (
        MediaFrameReader,
        MediaFrameReaderAcquisitionMode,
    )
    from winrt.windows.media.capture import MediaStreamType

    group = await _pick_source_group(display_name)
    settings = MediaCaptureInitializationSettings()
    settings.source_group = group
    settings.memory_preference = MediaCaptureMemoryPreference.CPU

    capture = MediaCapture()
    await capture.initialize_with_settings_async(settings)

    video_source = None
    for source in capture.frame_sources.values():
        if source.info.media_stream_type in (
            MediaStreamType.VIDEO_PREVIEW,
            MediaStreamType.VIDEO_RECORD,
        ):
            video_source = source
            break
    if video_source is None:
        values = list(capture.frame_sources.values())
        if values:
            video_source = values[0]

    if video_source is None:
        raise RuntimeError(f"No video source on device {display_name!r}")

    reader = await capture.create_frame_reader_async(video_source)
    reader.acquisition_mode = MediaFrameReaderAcquisitionMode.REALTIME
    status = await reader.start_async()
    from winrt.windows.media.capture.frames import MediaFrameReaderStartStatus

    if status != MediaFrameReaderStartStatus.SUCCESS:
        raise RuntimeError(f"MediaFrameReader start failed: {status}")
    return capture, reader, group


def _software_bitmap_to_bgr(sb) -> np.ndarray | None:
    from winrt.windows.graphics.imaging import (
        BitmapBufferAccessMode,
        BitmapPixelFormat,
        SoftwareBitmap,
    )

    if sb is None:
        return None
    if sb.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
        sb = SoftwareBitmap.convert(sb, BitmapPixelFormat.BGRA8)
    w = sb.pixel_width
    h = sb.pixel_height
    with sb.lock_buffer(BitmapBufferAccessMode.READ) as buffer:
        ref = buffer.create_reference()
        arr = np.frombuffer(bytes(ref), dtype=np.uint8)
    if arr.size < w * h * 4:
        return None
    bgra = arr[: w * h * 4].reshape((h, w, 4))
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


async def _acquire_bgr(reader, *, timeout_s: float = 2.0) -> np.ndarray | None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        frame = reader.try_acquire_latest_frame()
        if frame is not None:
            try:
                vmf = frame.video_media_frame
                if vmf is not None and vmf.software_bitmap is not None:
                    bgr = _software_bitmap_to_bgr(vmf.software_bitmap)
                    if bgr is not None:
                        return bgr
            finally:
                frame.close()
        await asyncio.sleep(0.01)
    return None


async def _shutdown(capture, reader) -> None:
    try:
        await reader.stop_async()
    except OSError:
        pass
    reader.close()
    capture.close()


class WinRTNativeCapture:
    """Native Windows MediaCapture; alias NativeWindowsCapture."""
    """read()/release() compatible with other VIGIL capture helpers."""

    def __init__(self, device_name: str = DEFAULT_DEVICE) -> None:
        self._device_name = device_name
        self._loop = asyncio.new_event_loop()
        self._capture = None
        self._reader = None
        self._group = None
        self._info: WinRTCaptureInfo | None = None
        try:
            self._capture, self._reader, self._group = self._loop.run_until_complete(
                _initialize_reader(device_name)
            )
            first = self._loop.run_until_complete(
                _acquire_bgr(self._reader, timeout_s=10.0)
            )
            if first is None:
                raise RuntimeError(
                    f"No frames from {device_name!r} within startup timeout"
                )
            h, w = first.shape[:2]
            self._info = WinRTCaptureInfo(
                device_name=device_name,
                device_id=self._group.id,
                source_group_id=self._group.id,
                actual_width=w,
                actual_height=h,
                pixel_format="BGRA8 to BGR",
            )
            self._pending: np.ndarray | None = first
        except Exception:
            self.release()
            raise

    @property
    def capture_info(self) -> WinRTCaptureInfo:
        if self._info is None:
            raise RuntimeError("Capture not initialized")
        return self._info

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._reader is None:
            return False, None
        if self._pending is not None:
            frame = self._pending
            self._pending = None
            return True, frame
        frame = self._loop.run_until_complete(
            _acquire_bgr(self._reader, timeout_s=1.0)
        )
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._capture is not None and self._reader is not None:
            try:
                self._loop.run_until_complete(
                    _shutdown(self._capture, self._reader)
                )
            except Exception:
                pass
        self._capture = None
        self._reader = None
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None


NativeWindowsCapture = WinRTNativeCapture


def open_winrt_capture(device_name: str) -> WinRTNativeCapture:
    return WinRTNativeCapture(device_name)


open_native_windows_capture = open_winrt_capture


def _cli_main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="WinRT MediaCapture connectivity test")
    parser.add_argument("--device-name", default=DEFAULT_DEVICE)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    try:
        cap = open_winrt_capture(args.device_name)
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    print_winrt_capture_info(cap.capture_info)
    ok_count = 0
    t0 = time.perf_counter()
    window = "VIGIL WinRT Preview"
    if args.preview:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        for i in range(args.frames):
            ok, frame = cap.read()
            if ok and frame is not None:
                ok_count += 1
                h, w = frame.shape[:2]
                print(f"Frame {i + 1}: OK {w}x{h}")
                if args.preview:
                    cv2.imshow(window, frame)
                    cv2.waitKey(1)
            else:
                print(f"Frame {i + 1}: FAIL")
        elapsed = time.perf_counter() - t0
        if ok_count > 1 and elapsed > 0:
            print(f"Measured FPS: {ok_count / elapsed:.2f}")
        print(f"Frames received: {ok_count}/{args.frames}")
    finally:
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
    sys.exit(0 if ok_count > 0 else 1)


if __name__ == "__main__":
    _cli_main()

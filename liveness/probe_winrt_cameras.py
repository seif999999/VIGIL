"""Enumerate video devices via native Windows WinRT APIs (no OpenCV/DirectShow)."""

from __future__ import annotations

import asyncio
import sys


async def main_async() -> int:
    from winrt.windows.devices.enumeration import DeviceInformation, DeviceClass
    from winrt.windows.media.capture.frames import MediaFrameSourceGroup

    print("=== DeviceInformation (VideoCapture) ===")
    devices = await DeviceInformation.find_all_async_device_class(
        DeviceClass.VIDEO_CAPTURE
    )
    for dev in devices:
        print(f"Name: {dev.name}")
        print(f"  Id: {dev.id}")
        print(f"  IsEnabled: {dev.is_enabled}")

    print("\n=== MediaFrameSourceGroup.FindAllAsync ===")
    groups = await MediaFrameSourceGroup.find_all_async()
    for grp in groups:
        print(f"DisplayName: {grp.display_name}")
        print(f"  Id: {grp.id}")
        for src in grp.source_infos:
            print(f"  Source: {src.id} type={src.source_kind} mediaStream={src.media_stream_type}")

    target = "Nazmy (Windows Virtual Camera)"
    found = any(d.name == target for d in devices) or any(
        g.display_name == target for g in groups
    )
    print(f"\nTarget '{target}' found: {found}")
    return 0 if found else 1


def main() -> None:
    try:
        code = asyncio.run(main_async())
    except ImportError as exc:
        print(f"[ERROR] WinRT packages required: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code)


if __name__ == "__main__":
    main()

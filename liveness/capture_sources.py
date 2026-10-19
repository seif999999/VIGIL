"""Mutually exclusive live capture source validation."""

from __future__ import annotations

import sys


def validate_exclusive_live_sources(
    *,
    windows_device: str | None = None,
    stream_url: str | None = None,
    dshow_device: str | None = None,
) -> None:
    sources = [
        name
        for name, val in (
            ("--windows-device", windows_device),
            ("--stream-url", stream_url),
            ("--dshow-device", dshow_device),
        )
        if val
    ]
    if len(sources) > 1:
        print(
            f"[ERROR] Use only one live source. Conflicting: {', '.join(sources)}",
            file=sys.stderr,
        )
        raise SystemExit(2)

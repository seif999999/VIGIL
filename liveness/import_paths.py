"""Keep liveness `detector` imports on liveness/detector.py (not face_detection)."""

from __future__ import annotations

import sys
from pathlib import Path

_LIVENESS_ROOT = Path(__file__).resolve().parent


def prepare_liveness_imports() -> None:
    root = str(_LIVENESS_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    cached = sys.modules.get("detector")
    if cached is not None and not hasattr(cached, "LivenessDetector"):
        del sys.modules["detector"]

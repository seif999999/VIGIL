"""Load face_detection.detector without clashing with liveness.detector."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

FACE_DETECTOR_PATH = (
    Path(__file__).resolve().parent.parent / "face_detection" / "detector.py"
)


def load_face_detector_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "vigil_face_detection_detector",
        FACE_DETECTOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load face detector from {FACE_DETECTOR_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

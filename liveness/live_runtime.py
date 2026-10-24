"""Shared live-session helpers: frame readiness, TF thread config, interrupt flag.

Does not change model weights, crops, thresholds, or auth decision rules.
"""

from __future__ import annotations

import os
import signal
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Startup / readiness states shown on HUD (not auth decisions).
CAMERA_INITIALIZING = "camera_initializing"
WAITING_FOR_FRAMES = "waiting_for_frames"
MODELS_LOADING = "models_loading"
FRAMES_OK = "frames_ok"
CAMERA_UNAVAILABLE = "unavailable"
AUTH_READY = "auth_ready"  # models ready + usable frames (informational)


@dataclass
class LatestFrame:
    frame: np.ndarray | None = None
    seq: int = 0
    capture_ts: float = 0.0


def configure_tf_threads_for_inference(*, intra: int | None = None, inter: int = 1) -> str:
    """
    Best-effort TensorFlow threading before models load.
    Returns a short status string (never raises).
    """
    notes: list[str] = []
    try:
        import cv2  # noqa: WPS433

        # Limit OpenCV BLAS/threads so they do not oversubscribe with TF/ORT.
        cv2.setNumThreads(1)
        notes.append("opencv_threads=1")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"opencv_threads_unavailable:{type(exc).__name__}")
    try:
        import tensorflow as tf  # noqa: WPS433

        cpus = os.cpu_count() or 4
        # Leave headroom for camera preview + SCRFD (onnxruntime) + OpenCV.
        intra_n = intra if intra is not None else max(2, min(6, max(2, cpus - 2)))
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        tf.config.threading.set_intra_op_parallelism_threads(intra_n)
        notes.append(f"tf_threads intra={intra_n} inter={inter}")
        return " ".join(notes)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"tf_threads_unavailable:{type(exc).__name__}")
        return " ".join(notes)


def frame_is_usable(frame: np.ndarray | None) -> bool:
    """
    Reject missing / empty / near-black / near-uniform buffers typical of
    camera backend startup. A textured empty scene (no face) still returns True
    so SCRFD can correctly emit NO_FACE.
    """
    if frame is None:
        return False
    if getattr(frame, "size", 0) == 0:
        return False
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        return False
    # Fast subsample — readiness only, not a biometric transform.
    sample = frame[::8, ::8, :3]
    mean = float(sample.mean())
    std = float(sample.std())
    if mean < 3.0 and std < 3.0:
        return False
    if std < 1.5:
        return False
    return True


class InterruptFlag:
    """Process-wide cooperative shutdown for Ctrl+C / SIGINT."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._installed = False
        self._last_signum: int | None = None

    @property
    def is_set(self) -> bool:
        return self._event.is_set()

    @property
    def last_signum(self) -> int | None:
        return self._last_signum

    def signal_reason(self) -> str:
        """Human-readable reason when the flag was set by a signal handler."""
        if self._last_signum is None:
            return "interrupt_flag"
        try:
            name = signal.Signals(self._last_signum).name
        except (ValueError, AttributeError):
            name = f"signal_{self._last_signum}"
        return f"signal:{name}"

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()
        self._last_signum = None

    def install(self) -> None:
        if self._installed:
            return

        def _handler(signum, frame) -> None:  # noqa: ARG001
            self._last_signum = int(signum)
            self._event.set()

        try:
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            # Not on main thread — KeyboardInterrupt still works in main.
            pass
        if hasattr(signal, "SIGBREAK"):
            try:
                signal.signal(signal.SIGBREAK, _handler)  # type: ignore[attr-defined]
            except (ValueError, OSError):
                pass
        self._installed = True


QUIT_KEY_CODES = frozenset({27, ord("q"), ord("Q")})


def poll_key_code(delay_ms: int = 1) -> int:
    """
    Return -1 when no key was pressed; otherwise the low 8 bits of the key code.

    Prefer this over ``waitKey(...) & 0xFF``, which turns OpenCV's no-key
    sentinel (-1) into 255 and obscures quit vs idle.
    """
    import cv2  # noqa: WPS433

    raw = int(cv2.waitKey(delay_ms))
    if raw == -1:
        return -1
    return raw & 0xFF


def safe_wait_key(delay_ms: int = 1) -> int:
    """cv2.waitKey wrapper; returns -1 when no key (does not mask -1 into 255)."""
    try:
        return poll_key_code(delay_ms)
    except KeyboardInterrupt:
        raise


def preview_window_closed(window: str) -> bool:
    """True when HighGUI reports the named window is gone / not visible."""
    import cv2  # noqa: WPS433

    try:
        # OpenCV docs: -1 when the window does not exist. Some Win32 builds
        # return 0.0 after destroy / user close instead of -1.
        prop = float(cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE))
    except Exception:  # noqa: BLE001
        return True
    return prop < 1.0


def run_with_interrupt_guard(fn: Callable[[], None], interrupt: InterruptFlag) -> None:
    """Run fn; on KeyboardInterrupt set interrupt flag instead of raw traceback spam."""
    try:
        fn()
    except KeyboardInterrupt:
        interrupt.set()

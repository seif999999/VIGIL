"""Startup frame gating, interrupt flag, and recorder close tests (no camera)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auth_decision import AuthDecision, AuthOutcome
from auth_pipeline import AuthPipelineResult
from fasnet_calibration_record import CalibrationSessionRecorder
from fasnet_liveness import FasNetClass, FasNetResult
from live_runtime import (
    CAMERA_UNAVAILABLE,
    FRAMES_OK,
    MODELS_LOADING,
    QUIT_KEY_CODES,
    WAITING_FOR_FRAMES,
    InterruptFlag,
    frame_is_usable,
    poll_key_code,
    preview_window_closed,
    safe_wait_key,
)
from live_friend_auth import (
    _camera_backend_candidates,
    _open_camera_validated,
    _read_capture_usability,
    draw_auth,
)


class FrameUsableTests(unittest.TestCase):
    def test_none_unusable(self) -> None:
        self.assertFalse(frame_is_usable(None))

    def test_black_startup_unusable(self) -> None:
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertFalse(frame_is_usable(black))

    def test_uniform_gray_unusable(self) -> None:
        gray = np.full((240, 320, 3), 10, dtype=np.uint8)
        self.assertFalse(frame_is_usable(gray))

    def test_textured_empty_scene_usable(self) -> None:
        rng = np.random.default_rng(1)
        frame = rng.integers(30, 200, (480, 640, 3), dtype=np.uint8)
        self.assertTrue(frame_is_usable(frame))


class InterruptTests(unittest.TestCase):
    def test_flag_set(self) -> None:
        flag = InterruptFlag()
        self.assertFalse(flag.is_set)
        flag.set()
        self.assertTrue(flag.is_set)

    def test_signal_reason_without_signum(self) -> None:
        flag = InterruptFlag()
        flag.set()
        self.assertEqual(flag.signal_reason(), "interrupt_flag")

    def test_clear_resets_signum(self) -> None:
        flag = InterruptFlag()
        flag._last_signum = 2
        flag.set()
        flag.clear()
        self.assertFalse(flag.is_set)
        self.assertIsNone(flag.last_signum)


class WaitKeyHelperTests(unittest.TestCase):
    def test_quit_key_codes(self) -> None:
        self.assertIn(27, QUIT_KEY_CODES)
        self.assertIn(ord("q"), QUIT_KEY_CODES)
        self.assertIn(ord("Q"), QUIT_KEY_CODES)
        # No-key sentinel must not be treated as quit after masking.
        self.assertNotIn((-1) & 0xFF, QUIT_KEY_CODES)

    def test_poll_key_code_no_key_is_minus_one(self) -> None:
        import cv2
        import numpy as np

        window = "vigil_test_waitkey"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        try:
            img = np.full((120, 160, 3), 40, dtype=np.uint8)
            cv2.imshow(window, img)
            key = poll_key_code(1)
            self.assertEqual(key, -1)
            self.assertEqual(safe_wait_key(1), -1)
        finally:
            cv2.destroyWindow(window)
            cv2.waitKey(1)

    def test_preview_window_closed_unknown_name(self) -> None:
        self.assertTrue(preview_window_closed("vigil_no_such_window_xyz"))


class HudStartupStateTests(unittest.TestCase):
    def test_draw_models_loading_no_crash(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # textured so it's a real buffer
        img[:] = 40
        draw_auth(
            img,
            None,
            stale=True,
            mode="continuous",
            busy=True,
            age_ms=None,
            camera_state=MODELS_LOADING,
            preview_fps=20.0,
            auth_hz=None,
        )

    def test_draw_waiting_frames(self) -> None:
        img = np.full((480, 640, 3), 50, dtype=np.uint8)
        draw_auth(
            img,
            None,
            stale=True,
            mode="continuous",
            busy=False,
            age_ms=None,
            camera_state=WAITING_FOR_FRAMES,
            preview_fps=None,
            auth_hz=None,
        )

    def test_stale_clears_accept_display_on_unavailable(self) -> None:
        img = np.full((480, 640, 3), 50, dtype=np.uint8)
        face = SimpleNamespace(x1=10, y1=10, x2=50, y2=50, confidence=0.9)
        det = SimpleNamespace(
            faces=(face,),
            face_count=1,
            decision=SimpleNamespace(value="FACE_OK"),
        )
        fas = FasNetResult(True, 0.9, FasNetClass.REAL, 1.0, (10, 10, 40, 40), None)
        outcome = AuthOutcome(
            AuthDecision.ACCEPT, "real_and_recognized", 1, "REAL", True, "Seif"
        )
        result = AuthPipelineResult(
            det, fas, None, outcome, 1.0, 1.0, 1.0, 3.0
        )
        # When camera unavailable path uses result=None in caller; draw must not
        # present ACCEPT as current for unavailable state with stale.
        draw_auth(
            img,
            result,
            stale=True,
            mode="continuous",
            busy=False,
            age_ms=5000.0,
            camera_state=CAMERA_UNAVAILABLE,
            preview_fps=None,
            auth_hz=None,
        )


class RecorderCloseOnceTests(unittest.TestCase):
    def test_close_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rec = CalibrationSessionRecorder.create(
                label="REAL", camera_index=0, out_dir=Path(td)
            )
            p1 = rec.close()
            p2 = rec.close()
            self.assertEqual(p1, p2)
            ends = [
                json.loads(line)
                for line in p1.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(sum(1 for x in ends if x.get("type") == "session_end"), 1)


class ConstantsTests(unittest.TestCase):
    def test_frames_ok_value(self) -> None:
        self.assertEqual(FRAMES_OK, "frames_ok")


class _FakeCapture:
    """Minimal VideoCapture stand-in for open/fallback unit tests."""

    def __init__(self, frames: list[np.ndarray], *, opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self._i = 0
        self.released = False

    def isOpened(self) -> bool:
        return self._opened and not self.released

    def read(self):
        if not self.isOpened() or not self._frames:
            return False, None
        idx = min(self._i, len(self._frames) - 1)
        self._i += 1
        return True, self._frames[idx]

    def release(self) -> None:
        self.released = True
        self._opened = False


def _textured_frame() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(30, 220, (240, 320, 3), dtype=np.uint8)


def _black_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


class CameraOpenFallbackTests(unittest.TestCase):
    def test_backend_order_prefers_msmf_then_dshow(self) -> None:
        names = [n for n, _ in _camera_backend_candidates()]
        self.assertEqual(names[-1], "ANY")
        if "MSMF" in names:
            self.assertEqual(names[0], "MSMF")
        if "MSMF" in names and "DSHOW" in names:
            self.assertLess(names.index("MSMF"), names.index("DSHOW"))

    def test_read_usability_rejects_all_black(self) -> None:
        cap = _FakeCapture([_black_frame()] * 8)
        ok_n, usable_n, mean, std, _shape = _read_capture_usability(cap, n_reads=8)
        self.assertEqual(ok_n, 8)
        self.assertEqual(usable_n, 0)
        self.assertEqual(mean, 0.0)
        self.assertEqual(std, 0.0)

    def test_read_usability_accepts_textured(self) -> None:
        cap = _FakeCapture([_textured_frame()] * 4)
        ok_n, usable_n, _mean, _std, _shape = _read_capture_usability(cap, n_reads=4)
        self.assertEqual(ok_n, 4)
        self.assertGreaterEqual(usable_n, 1)

    def test_fallback_skips_black_backend(self) -> None:
        black = _FakeCapture([_black_frame()] * 8)
        good = _FakeCapture([_textured_frame()] * 8)
        sleeps: list[float] = []

        def open_fn(_index: int, api: int | None):
            # First backend (api != None) returns black; second succeeds.
            if api is not None and not black.released:
                return black
            return good

        cap = _open_camera_validated(
            0,
            [("DSHOW", 700), ("MSMF", 1400)],
            open_fn=open_fn,
            sleep_fn=sleeps.append,
            validate_reads=8,
            min_usable=1,
            settle_s=0.25,
        )
        self.assertIs(cap, good)
        self.assertTrue(black.released)
        self.assertFalse(good.released)
        self.assertEqual(sleeps, [0.25])

    def test_all_backends_black_raises_safe_failure(self) -> None:
        caps = [
            _FakeCapture([_black_frame()] * 4),
            _FakeCapture([_black_frame()] * 4),
        ]
        idx = {"i": 0}

        def open_fn(_index: int, _api: int | None):
            cap = caps[idx["i"]]
            idx["i"] += 1
            return cap

        with self.assertRaises(RuntimeError) as ctx:
            _open_camera_validated(
                0,
                [("MSMF", 1400), ("DSHOW", 700)],
                open_fn=open_fn,
                sleep_fn=lambda _s: None,
                validate_reads=4,
                min_usable=1,
                settle_s=0.1,
            )
        self.assertIn("Cannot open usable camera", str(ctx.exception))
        self.assertTrue(caps[0].released)
        self.assertTrue(caps[1].released)


if __name__ == "__main__":
    unittest.main()
